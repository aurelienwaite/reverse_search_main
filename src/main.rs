use anyhow::{anyhow, Result};
use clap::Parser;
use itertools::Itertools;
use log::{debug, info};
use ndarray::prelude::*;
use parquet::{
    basic::{Repetition, Type as PhysicalType},
    column::writer::ColumnWriter,
    errors::ParquetError,
    file::{
        properties::WriterProperties,
        reader::{FileReader, SerializedFileReader},
        writer::SerializedFileWriter,
    },
    record::RowAccessor,
    schema::types::Type,
};
use rayon::prelude::*;
use reverse_search::guided_search::{Guide, Scorer};
use reverse_search::{reverse_search, Polytope, ReverseSearchOut, Searcher};
use reverse_search_main::{runner, Accuracy, MeanAverageError, ThreadPool};
use serde::{Deserialize, Serialize};
use simplelog::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{prelude::*, BufReader, BufWriter},
    rc::Rc,
    string::String,
    sync::Arc,
    vec::Vec,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    polytope_file: String,

    #[arg(long)]
    labels: Option<String>,

    #[arg(long)]
    responses: Option<String>,

    #[arg(long)]
    clusters: Option<String>,

    #[arg(long)]
    polytope_out: String,

    #[arg(long)]
    reverse_search_out: String,

    #[arg(long)]
    num_states: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct InputPolytope {
    vertices: Vec<Vec<f64>>,
}

#[derive(Serialize, Debug)]
struct Output<'a> {
    minkowski_decomp: &'a Vec<usize>,
    score: f64,
}

impl InputPolytope {
    fn to_array_poly(&self) -> Result<Polytope> {
        let dim = self.vertices.first().ok_or(anyhow!("No polytopes!"))?.len();
        let mut array = Array2::<f64>::zeros((self.vertices.len(), dim));
        for i in 0..self.vertices.len() {
            for j in 0..dim {
                array[[i, j]] = self.vertices[i][j];
            }
        }
        Ok(Polytope::new(array))
    }
}

fn make_scorer<'a>(scorer: impl Scorer + 'a) -> Rc<dyn Scorer + 'a> {
    Rc::new(scorer)
}

fn populate_string(filename: String) -> Result<String> {
    let json_file = File::open(filename)?;
    let mut buf_reader = BufReader::new(json_file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    Ok(contents)
}

fn read_polytope(polytope_filename: String) -> Result<Vec<Polytope>> {
    let contents = populate_string(polytope_filename)?;
    let deserialised: Vec<InputPolytope> = serde_json::from_str(&contents)?;
    info!("Loaded {} polytopes", deserialised.len());
    let poly = deserialised
        .iter()
        .map(|p| p.to_array_poly())
        .collect::<Result<Vec<_>>>()?;
    Ok(poly)
}

fn read_labels(labels_filename: String) -> Result<Vec<usize>> {
    let contents = populate_string(labels_filename)?;
    let deserialised: Vec<usize> = serde_json::from_str(&contents)?;
    Ok(deserialised)
}

fn read_responses(responses_filename: String) -> Result<(Vec<usize>, Array1<f64>)> {
    let responses_file = File::open(responses_filename)?;
    let reader = SerializedFileReader::new(responses_file)?;
    let parquet_metadata = reader.metadata();
    let fields = parquet_metadata.file_metadata().schema().get_fields();
    let mut cluster_id_pos_opt: Option<usize> = None;
    let mut response_pos_opt: Option<usize> = None;
    for (pos, column) in fields.iter().enumerate() {
        match column.name() {
            "Cluster_ID" => Ok(cluster_id_pos_opt = Some(pos)),
            "Response" => Ok(response_pos_opt = Some(pos)),
            unknown => Err(anyhow!("Unknown column: {}", unknown)),
        }?
    }
    let cluster_id_pos = cluster_id_pos_opt.ok_or(anyhow!("No cluster id in schema"))?;
    let response_pos = response_pos_opt.ok_or(anyhow!("No response in schema"))?;

    let mut labels: Vec<usize> = Vec::new();
    let mut responses: Vec<f64> = Vec::new();
    let mut iter = reader.get_row_iter(None)?;
    while let Some(record_res) = iter.next() {
        let record = record_res?;
        labels.push(record.get_int(cluster_id_pos)?.try_into()?);
        responses.push(record.get_double(response_pos)?);
    }
    Ok((labels, Array::from_vec(responses)))
}

fn read_clusters(responses_filename: String) -> Result<Array1<f64>> {
    let responses_file = File::open(responses_filename)?;
    let reader = SerializedFileReader::new(responses_file)?;
    let parquet_metadata = reader.metadata();
    let fields = parquet_metadata.file_metadata().schema().get_fields();
    let mut cluster_id_pos_opt: Option<usize> = None;
    let mut cluster_val_pos_opt: Option<usize> = None;
    for (pos, column) in fields.iter().enumerate() {
        match column.name() {
            "cluster_id" => Ok(cluster_id_pos_opt = Some(pos)),
            "cluster_value" => Ok(cluster_val_pos_opt = Some(pos)),
            unknown => Err(anyhow!("Unknown column: {}", unknown)),
        }?
    }
    let cluster_id_pos = cluster_id_pos_opt.ok_or(anyhow!("No cluster id in schema"))?;
    let response_pos = cluster_val_pos_opt.ok_or(anyhow!("No response in schema"))?;

    let mut clusters: HashMap<usize, f64> = HashMap::new();
    let mut iter = reader.get_row_iter(None)?;
    while let Some(record_res) = iter.next() {
        let record = record_res?;
        let label: usize = record.get_int(cluster_id_pos)?.try_into()?;
        clusters.insert(label, record.get_double(response_pos)?);
    }
    let mut cluster_ary = Array1::<f64>::zeros(clusters.len());
    for (cluster_id, cluster_val) in clusters {
        cluster_ary[cluster_id] = cluster_val;
    }
    Ok(cluster_ary)
}

fn read_parquet_polytope(poly_str: String) -> Result<Vec<Polytope>> {
    // This is not very robust code. We assume that there are the same number of vertices
    // in each polytope, and that the data is ascending order with respect to polytope, vertex, and dim

    let poly_file = File::open(poly_str)?;
    let reader = SerializedFileReader::new(poly_file)?;
    let parquet_metadata = reader.metadata();
    let fields = parquet_metadata.file_metadata().schema().get_fields();
    let mut fields_map: HashMap<&str, usize> = HashMap::new();
    for (pos, column) in fields.iter().enumerate() {
        let name = column.name();
        fields_map.insert(name, pos);
    }

    let extract = |k: &str| fields_map.get(k).ok_or(anyhow!("No {} col", k)).map(|i| *i);
    let col_pos = ["polytope", "vertex", "dim", "value"]
        .into_iter()
        .map(extract)
        .collect::<Result<Vec<_>>>()?;
    let [p_pos, v_pos, d_pos, va_pos] = col_pos.as_slice() else {
        panic!("Can't happen")
    };

    let (mut max_p, mut max_v, mut max_d) = (0_usize, 0_usize, 0_usize);

    let mut values: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut iter = reader.get_row_iter(None)?;
    while let Some(record_res) = iter.next() {
        let record = record_res?;
        let p_value: usize = record.get_int(*p_pos)?.try_into()?;
        max_p = max_p.max(p_value);
        let v_value: usize = record.get_int(*v_pos)?.try_into()?;
        max_v = max_v.max(v_value);
        let d_value: usize = record.get_int(*d_pos)?.try_into()?;
        max_d = max_d.max(d_value);
        let datum: f64 = record.get_double(*va_pos)?;
        values.push((p_value, v_value, d_value, datum));
    }
    debug!("Max p: {}, max v {}, max d{}", max_p, max_v, max_d);
    let mut raw_arrays: Vec<Array2<f64>> = Vec::new();
    for _p in 0..(max_p + 1) {
        let vertices = Array2::<f64>::zeros((max_v + 1, max_d + 1));
        raw_arrays.push(vertices);
    }
    for (polytope, vertex, dim, datum) in values{
        raw_arrays[polytope][[vertex, dim]] = datum;
    }
    let polytopes = raw_arrays.into_iter().map(Polytope::new).collect_vec();
    Ok(polytopes)
}

/*
fn write_polytope(poly_str: &Vec<FullPolytope>, out_filename: &String) -> Result<()>{
    info!("Saving {} polytopes to {}", poly_str.len(), out_filename);
    let out_string = serde_json::to_string_pretty(poly_str)?;
    let poly_out_file = File::create(out_filename)?;
    let mut writer = BufWriter::new(poly_out_file);
    writer.write(out_string.as_bytes())?;
    return Ok(());
}*/

fn run_guided_search(
    search: &Searcher,
    guide: &Guide,
    mut writer_callback: Box<impl FnMut(ReverseSearchOut) -> Result<()>>,
    num_states: Option<usize>,
) -> Result<()> {
    info!("Starting search");
    let thread_pool = ThreadPool::new(4, search.clone());
    let runner_res = runner(&thread_pool, &guide, 10)?;
    let mut count = 0usize;
    for outputs in runner_res {
        for output in outputs? {
            writer_callback(output)?;
            count += 1;
        }
        if let Some(bound) = num_states {
            if count >= bound {
                break;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Debug,
        Config::default(),
        TerminalMode::Stdout,
        ColorChoice::Auto,
    )?;

    let args = Args::parse();

    info!("Loading {}!", args.polytope_file);

    let suffix = args
        .polytope_file
        .split(".")
        .last()
        .ok_or(anyhow!("No file suffix {}", args.polytope_file))?;
    let poly = match suffix {
        "json" => read_polytope(args.polytope_file),
        "jsn" => read_polytope(args.polytope_file),
        "parquet" => read_parquet_polytope(args.polytope_file),
        _ => Err(anyhow!("Unknown suffix {}", suffix)),
    }?;

    info!("Saving search results to {}", &args.reverse_search_out);
    let states_out_jsonl = format!("{}.jsonl", args.reverse_search_out);
    let states_out_file = File::create(&states_out_jsonl)?;
    let mut writer = BufWriter::new(states_out_file);

    let states_out_parquet = format!("{}.parquet", args.reverse_search_out);
    let params_file = File::create(&states_out_parquet)?;
    let dim = poly
        .first()
        .ok_or(anyhow!("No polytopes!"))?
        .vertices
        .shape()[1];
    let fields = (0..dim)
        .into_iter()
        .map(|i| {
            Type::primitive_type_builder(format!("parameter_{}", i).as_str(), PhysicalType::DOUBLE)
                .with_repetition(Repetition::REQUIRED)
                .build()
        })
        .collect::<Result<Vec<_>, ParquetError>>()?;
    let schema = Arc::new(
        Type::group_type_builder("schema")
            .with_fields(fields.into_iter().map(|f| Arc::new(f)).collect_vec())
            .build()?,
    );
    let props = Arc::new(WriterProperties::builder().build());
    let mut parquet_writer = SerializedFileWriter::new(params_file, schema, props)?;

    let labels = args.labels.map(read_labels).transpose()?;
    let responses = args.responses.map(read_responses).transpose()?;
    let clusters = args.clusters.map(read_clusters).transpose()?;

    let scorer: Option<Rc<dyn Scorer>> = match (labels, responses, clusters) {
        (Some(inner_labels), None, None) => Ok(Some(make_scorer(Accuracy::new(inner_labels)))),
        (None, Some((_, responses)), Some(cluster_ids)) => Ok(Some(make_scorer(
            MeanAverageError::new(cluster_ids, responses),
        ))),
        (None, None, None) => Ok(None),
        _ => Err(anyhow!("Need to supply labels, or clusters and responses")),
    }?;

    
    info!("Filling polytopes");
    let mut filled = poly.into_par_iter().map(|mut p|  {
        p.fill().map(|_| p)
    }).collect::<Result<Vec<_>>>()?;
    

    let mut counter = 0;
    match scorer {
        Some(scorer) => {
            let search = Searcher::setup_reverse_search(&filled)?;
            let guide = Guide::new(&search, scorer.clone(), Some(42))?;
            let writer_callback = Box::new(|rs_out: ReverseSearchOut| {
                counter += 1;
                info!("Writing guided result {}", counter);
                {
                    let mut row_group_writer = parquet_writer.next_row_group()?;
                    for param in &rs_out.param {
                        let col_writer = row_group_writer.next_column()?;
                        if let Some(mut inner_col_writer) = col_writer {
                            if let ColumnWriter::DoubleColumnWriter(ref mut typed) =
                                inner_col_writer.untyped()
                            {
                                let values = vec![*param];
                                let written = typed.write_batch(&values, None, None)?;
                                assert!(written == 1, "written was {}", written);
                            } else {
                                return Err(anyhow!("Unexpected column writer"));
                            }
                            inner_col_writer.close()?;
                        } else {
                            return Err(anyhow!("No column writer"));
                        }
                    }
                    row_group_writer.close()?;
                }
                let decomp =  guide.map_polytope_indices(rs_out.minkowski_decomp_iter()).collect_vec();
                let accuracy = -1. * scorer.score(Box::new(decomp.iter().map(|v| *v)));
                let output = Output {
                    minkowski_decomp: &decomp,
                    score: accuracy,
                };
                let out_string = serde_json::to_string(&output)? + "\n";
                writer.write(out_string.as_bytes())?;
                writer.flush()?;
                return Ok(());
            });
            run_guided_search(&search, &guide, writer_callback, args.num_states)?;
            parquet_writer.close()?;
            Ok(())
        }
        None => {
            let writer_callback = Box::new(|rs_out: ReverseSearchOut| {
                counter += 1;
                info!("Writing result {}", counter);
                let decomp = rs_out.minkowski_decomp_iter().collect_vec();
                let output = Output {
                    minkowski_decomp: &decomp,
                    score: 1.,
                };
                let out_string = serde_json::to_string(&output)? + "\n";
                writer.write(out_string.as_bytes())?;
                writer.flush()?;
                return Ok(());
            });

            reverse_search(&mut filled, writer_callback)?;
            //write_polytope(&poly, &args.polytope_out)?;
            parquet_writer.close()?;
            Ok(())
        }
    }
}
