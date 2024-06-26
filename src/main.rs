use anyhow::{anyhow, Result};
use clap::Parser;
use core::future::Future;
use futures::future::BoxFuture;
use itertools::Itertools;
use log::{debug, info};
use ndarray::prelude::*;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use reverse_search::guided_search::Guide;
use reverse_search::{reverse_search, Polytope, ReverseSearchOut, Searcher, StepResult, TreeIndex};
use reverse_search_main::{runner, ThreadPool};
use serde::{Deserialize, Serialize};
use simplelog::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::pin::{pin, Pin};
use std::rc::Rc;
use std::string::String;
use std::vec::Vec;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    polytope_file: String,

    #[arg(long)]
    labels: Option<String>,

    #[arg(long)]
    polytope_out: String,

    #[arg(long)]
    reserve_search_out: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct InputPolytope {
    vertices: Vec<Vec<f64>>,
}

#[derive(Serialize, Debug)]
struct Output<'a> {
    param: &'a Array1<f64>,
    minkowski_decomp: &'a Vec<usize>,
    accuracy: f32,
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

    let mut values: Vec<f64> = Vec::new();
    let mut iter = reader.get_row_iter(None).unwrap();
    while let Some(record_res) = iter.next() {
        let record = record_res?;
        max_p = max_p.max(record.get_int(*p_pos)?.try_into()?);
        max_v = max_v.max(record.get_int(*v_pos)?.try_into()?);
        max_d = max_d.max(record.get_int(*d_pos)?.try_into()?);
        values.push(record.get_double(*va_pos)?);
    }
    let mut polytopes: Vec<Polytope> = Vec::new();
    for p in 0..(max_p + 1) {
        let mut vertices = Array2::<f64>::zeros((max_v + 1, max_d + 1));
        for v in 0..(max_v + 1) {
            for d in 0..(max_d + 1) {
                vertices[[v, d]] = values[p * (max_v + 1) * (max_d + 1) + v * (max_d + 1) + d]
            }
        }
        //info!("vertices {:?}", vertices);
        polytopes.push(Polytope::new(vertices));
    }

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
    polys: &mut [Polytope],
    labels: &[usize],
    mut writer_callback: Box<impl FnMut(ReverseSearchOut) -> Result<()>>,
) -> Result<()> {
    info!("Filling polytopes");
    for poly in &mut *polys {
        poly.fill()?;
    }
    info!("Starting search");
    let search = Searcher::setup_reverse_search(polys)?;
    let thread_pool = ThreadPool::new(4, search.clone());
    let guide = Guide::new(&search, labels, Some(42))?;
    let runner_res = runner(&thread_pool, &guide, 1000)?;
    let mut count = 0usize;
    for outputs in runner_res {
        for output in outputs? {
            writer_callback(output)?;
            count += 1;
        }
        if count > 1000 {
            break;
        }
        
    }
    Ok(())
}

fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
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
    let mut poly = match suffix {
        "json" => read_polytope(args.polytope_file),
        "jsn" => read_polytope(args.polytope_file),
        "parquet" => read_parquet_polytope(args.polytope_file),
        _ => Err(anyhow!("Unknown suffix {}", suffix)),
    }?;

    info!("Saving search results to {}", &args.reserve_search_out);
    let states_out_file = File::create(&args.reserve_search_out)?;
    let mut writer = BufWriter::new(states_out_file);

    let mut counter = 0;
    let labels = args.labels.map(read_labels).transpose()?;
    match labels {
        Some(labels) => {
            let writer_callback = Box::new(|rs_out: ReverseSearchOut| {
                counter += 1;
                info!("Writing guided result {}", counter);
                let decomp = rs_out.minkowski_decomp_iter().collect_vec();
                let matches: u32 = labels.iter().zip_eq(&decomp).map(|(l, d)| {
                    if *l == *d {
                        1u32
                    }else {
                        0u32
                    }
                }).sum();
                let output = Output {
                    param: &rs_out.param,
                    minkowski_decomp: &decomp,
                    accuracy: matches as f32 / labels.len() as f32,
                };
                let out_string = serde_json::to_string(&output)? + "\n";
                writer.write(out_string.as_bytes())?;
                writer.flush()?;
                return Ok(());
            
        });
        run_guided_search(&mut poly, &labels, writer_callback)
        }
        None => {
            let writer_callback = Box::new(|rs_out: ReverseSearchOut| {
                counter += 1;
                info!("Writing result {}", counter);
                let accuracy = None
                    .map(|l: &[usize]| {
                        1.
                        //accuracy(&rs_out.minkowski_decomp, l)
                    })
                    .unwrap_or(-1.);
                let decomp = rs_out.minkowski_decomp_iter().collect_vec();
                let output = Output {
                    param: &rs_out.param,
                    minkowski_decomp: &decomp,
                    accuracy: accuracy,
                };
                let out_string = serde_json::to_string(&output)? + "\n";
                writer.write(out_string.as_bytes())?;
                writer.flush()?;
                return Ok(());
            });

            reverse_search(&mut poly, writer_callback)?;
            //write_polytope(&poly, &args.polytope_out)?;
            Ok(())
        }
    }
}
