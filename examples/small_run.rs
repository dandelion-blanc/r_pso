extern crate rayon;
extern crate tokio;
extern crate pso;

use pso::opt_bench;
use pso::prelude::{*};
use tokio::spawn;
use rayon::prelude::*;

use core::f64;
use std::{fmt, ops::DerefMut};
use std::fmt::*;
use std::mem;

use rand::*;
use rand::prelude::*;

use gnuplot::*;


#[tokio::main]
async fn main()
{
    rayon::ThreadPoolBuilder::new()
    //.num_threads(4)
    .build_global()
    .unwrap();

    let max_t = 2_usize;
    let mut best_mem = Vec::<f64>::with_capacity(100);

    let mut pso = Origin::default()
        .set_particles(&100)
        .set_variables(&10)
        .set_max_iter(&100000)
        .set_ofunc(opt_bench::sphere)
		.set_init_func(Particle::default_init);


    for _ in 0..max_t
    {
        let result = pso.optimeze();

        println!("init {:?}", pso.update_vals.get(0).unwrap());
		println!("init {:?}", pso.update_value.get(0).unwrap());
        println!("optimized { }", pso.gbest.clone().unwrap());
        best_mem.push(pso.gbest.clone().unwrap().fvalue().unwrap());
    }

    let minimam = best_mem.iter().fold(0.0/0.0, |m, v| v.min(m));
    println!("{ }", minimam);
}
