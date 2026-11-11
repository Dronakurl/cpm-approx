//! Reads an `analyze` request (JSON, as the Python `pykhtin` library) from stdin or the file given
//! as first argument and prints the response as JSON. `RAYON_NUM_THREADS` limits the parallelism.
use std::io::Read;

fn main() {
    let mut input = String::new();
    let res = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(path).map(|s| input = s),
        None => std::io::stdin().read_to_string(&mut input).map(|_| ()),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
    let out = serde_json::from_str::<pykhtin::Request>(&input)
        .map_err(|e| e.to_string())
        .and_then(|r| pykhtin::analyze(&r));
    match out {
        Ok(r) => println!("{}", serde_json::to_string(&r).expect("serialise")),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
