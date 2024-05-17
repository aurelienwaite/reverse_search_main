use anyhow::{anyhow, Result};
use clap::Parser;
use log::{info, debug};
use ndarray::prelude::*;
use reverse_search::{reverse_search, Polytope, ReverseSearchOut};
use serde::{Deserialize, Serialize};
use simplelog::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::string::String;
use std::vec::Vec;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    polytope_file: String,

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
struct Output<'a>{
    param: &'a Array1<f64>,
    minkowski_decomp: &'a Vec<usize>
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

fn read_polytope(poly_str: String) -> serde_json::Result<Vec<InputPolytope>> {
    let deserialised: Vec<InputPolytope> = serde_json::from_str(&poly_str)?;
    info!("Loaded {} polytopes", deserialised.len());
    Ok(deserialised)
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

fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    )?;

    let args = Args::parse();

    info!("Loading {}!", args.polytope_file);

    let json_file = File::open(args.polytope_file)?;
    let mut buf_reader = BufReader::new(json_file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    let input_poly = read_polytope(contents)?;
    let mut poly = input_poly
        .iter()
        .map(|p| p.to_array_poly())
        .collect::<Result<Vec<_>>>()?;

    debug!("Polys {:?}", poly);
    info!("Saving search results to {}", &args.reserve_search_out);
    let states_out_file = File::create(&args.reserve_search_out)?;
    let mut writer = BufWriter::new(states_out_file);

    let mut counter = 0;
    let writer_callback = Box::new(|rs_out: ReverseSearchOut| {
        counter += 1;
        info!("Writing result {}", counter);
        let output = Output{
            param: &rs_out.param,
            minkowski_decomp: &rs_out.minkowski_decomp
        };
        let out_string = serde_json::to_string(&output)? + "\n";
        writer.write(out_string.as_bytes())?;
        return Ok(());
    });

    reverse_search(&mut poly, writer_callback)?;
    //write_polytope(&poly, &args.polytope_out)?;
    return Ok(());
}
