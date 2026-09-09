//! Print what this machine is and what it can do.
//!
//! `cargo run --release --example probe -p neurax-probe`
//!
//! Release matters: the kernels in `bench` are ordinary Rust loops, and an
//! unoptimised build measures the debug build rather than the hardware — off
//! by an order of magnitude, in the direction that would make every design
//! look far too slow.
fn main() {
    let mut profile = neurax_probe::detect();
    println!("{}", serde_json::to_string_pretty(&profile).unwrap());

    eprintln!("\nmeasuring…");
    profile.compute = Some(neurax_probe::measure_compute(&profile.cpu.model));
    println!("{}", serde_json::to_string_pretty(&profile.compute).unwrap());
}
