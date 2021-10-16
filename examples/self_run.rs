extern crate rayon;
extern crate tokio;
extern crate pso;

use pso::opt_bench;
use pso::prelude::{*};
use tokio::spawn;
use rayon::prelude::*;

use std::f64;
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

    let max_t = 100000_usize;
    let mut best_mem = Vec::<f64>::with_capacity(100);

    let mut pso = Origin::default()
        .set_particles(&100)
        .set_variables(&10)
        .set_max_iter(&(max_t as u64))
        .set_ofunc(Some(opt_bench::sphere));


    for _ in 0..100{
        pso.initialize();
        pso.update_swarm();
        println!("init : {:?}", pso.gbest);
    
    for t in 0..max_t
    {
        let ofunc = pso.ofunc().unwrap();
        let mut particles;
        {
            let st_particles = pso.particles_mut();
            particles = mem::replace(st_particles, None).unwrap();
        }
        let gbest;
        {
            gbest = mem::replace(&mut pso.gbest, None).unwrap();
        }
        {
            let gbest = gbest.x();

            particles.par_iter_mut()
                .for_each(
                    |p|
                    {
                        pso.particle_transfer(p, &gbest);
                        p.update();
                    });
        }
        {
            let st_particles = pso.particles_mut();
            let _ = mem::replace(st_particles, Some(particles));
        }
        {
            let _ = mem::replace(&mut pso.gbest, Some(gbest));
        }
        pso.update_swarm();
    }
    println!("optimized { }", pso.gbest.clone().unwrap());
    best_mem.push(pso.gbest.clone().unwrap().fvalue().unwrap());
    }
    let minimam = best_mem.iter().fold(0.0/0.0, |m, v| v.min(m));
    println!("{ }", minimam);
}
