use super::*;

use std::sync::Arc;
use std::iter;

use crate::rand::*;
use crate::rand::prelude::*;
use crate::rand_distr::StandardNormal;

#[derive(Clone)]
pub struct LimitedPso<O, I>
{
	// optimization variables
	pub particles :Option<Vec<Particle>>,
	pub gbest :Option<SearchPoint>,

	pub(crate) n_swarms :usize,
	pub(crate) n_particles :usize,
	pub(crate) n_variables :usize,

	pub(crate) w :f64,
	pub(crate) step_w :f64,
	pub(crate) w0 :f64,
	pub(crate) wt :f64,
	pub(crate) c1 :f64,
	pub(crate) c2 :f64,

	pub func :Option<Arc<O>>,
	pub init_func :Option<Arc<I>>,

	pub update_value :Vec<f64>,
	pub update_evals :Vec<u64>,
	pub update_vals :Vec<Vec<f64>>,
	pub search_vals :Vec<Vec<Vec<f64>>>,

	// parameter variables
	pub(crate) max_evaluations :u64,
	pub(crate) max_iter :u64,
	pub(crate) current_iter :u64,
	pub(crate) max_run :usize,

	pub(crate) upper_x :f64,
	pub(crate) lower_x :f64,
	pub(crate) init_ux :f64,
	pub(crate) init_lx :f64,
	pub(crate) upper_v :f64,
	pub(crate) lower_v :f64,
	pub(crate) init_uv :f64,
	pub(crate) init_lv :f64,

	pub(crate) sigma :f64,
}
impl<O, I> LimitedPso<O, I>
where
	O :Fn(&[f64]) -> f64,
	I :Fn(usize) -> Vec<f64>
{
	pub fn new() -> Self
	{
		Self
		{
			particles :None,
			gbest :None,

			n_swarms :0,
			n_particles :0,
			n_variables :0,

			w :0.0,
			step_w:0.0,
			w0 :0.0,
			wt :0.0,
			c1 :0.0,
			c2 :0.0,

			func :None,
			init_func: None,

			update_value: Vec::new(),
			update_evals: Vec::new(),
		    update_vals: Vec::new(),
			search_vals: Vec::new(),

			max_evaluations :0,
			max_iter :0,
			current_iter :0,
			max_run :0,

			upper_x :0.0,
			lower_x :0.0,
			init_ux :0.0,
			init_lx :0.0,

			upper_v :0.0,
			lower_v :0.0,
			init_uv :0.0,
			init_lv :0.0,

			sigma :0.0
		}
	}

	pub fn set_c(mut self, c1 :&f64, c2 :&f64) -> Self
	{
		self.c1 = *c1;
		self.c2 = *c2;
		self
	}

	pub fn set_init_lx(mut self, init_lx :&f64) -> Self
	{
		self.init_lx = *init_lx;
		self
	}

	pub fn set_init_ux(mut self, init_ux :&f64) -> Self
	{
		self.init_ux = *init_ux;
		self
	}

	pub fn set_init_xrange(mut self, init_lx :&f64, init_ux :&f64) -> Self
	{
		self.init_ux = *init_ux;
		self.init_lx = *init_lx;

		self
	}

	pub fn set_lx(mut self, lx :&f64) -> Self
	{
		self.lower_x = *lx;
		self
	}

	pub fn set_max_evaluations(mut self, max_evaluations :&u64) -> Self
	{
		self.max_evaluations = *max_evaluations;
		self
	}

	pub fn set_max_iter(mut self, max_iter :&u64) -> Self
	{
		self.max_iter = *max_iter;
		self
	}

	pub fn set_max_run(mut self, max_run :&usize) -> Self
	{
		self.max_run = *max_run;
		self
	}
	pub fn set_ofunc(mut self, ofunc :O) -> Self
	{
		self.func = Some(Arc::new(ofunc));
		self
	}

	pub fn set_ofunc_arc(mut self, ofunc :Option<Arc<O>>) -> Self
	{
		self.func = ofunc;
		self
	}
	pub fn set_init_func(mut self, func :I) -> Self
	{
		self.init_func = Some(Arc::new(func));
		self
	}

	pub fn set_init_func_arc(mut self, func :Option<Arc<I>>) -> Self
	{
		self.init_func = func;
		self
	}

	pub fn set_particles(mut self, n_particles :&usize) -> Self
	{
		self.n_particles = *n_particles;
		self
	}

	pub fn set_swarms(mut self, n_swarms :&usize) -> Self
	{unimplemented!();
		self.n_swarms = *n_swarms;
		self
	}

	pub fn set_ux(mut self, ux :&f64) -> Self
	{
		self.upper_x = *ux;
		self
	}

	pub fn set_variables(mut self, n_variables :&usize) -> Self
	{
		self.n_variables = *n_variables;
		self
	}

	pub fn set_wrange(mut self, w0 :&f64, wt :&f64) -> Self
	{
		self.w0 = *w0;
		self.wt = *wt;
		self
	}

	pub fn set_xrange(mut self, lx :&f64, ux :&f64) -> Self
	{
		self.upper_x = *ux;
		self.lower_x = *lx;
		self
	}

	pub fn set_sigma(mut self, sigma :f64) -> Self
	{
		self.sigma = sigma;
		self
	}
}
impl<O, I> Pso<Particle, SearchPoint, PsoResult, O, I> for LimitedPso<O, I>
where
	O :Fn(&[f64]) -> f64 + Sync + Send,
	I :Fn(usize) -> Vec<f64> + Sync + Send
{
	fn initialize(&mut self)
	{
		use std::mem;

		self.check_parameters();			// all parameters assertion
		{// particles initialize
			let range = rand::distributions::Uniform::new(self.init_xrange().0, self.init_xrange().1);
			let mut rng = thread_rng();
			let n_variables  = &self.n_variables();
			let n_particles = &self.n_particles();
			let sigma = self.sigma;
			let ofunc = self.ofunc().as_ref().unwrap().clone();
			let ifunc = self.init_func().as_ref().unwrap().clone();
			let st_particles = self.particles_vec();
			let mut particles = mem::replace(st_particles, None).unwrap();
				particles.clear();
				particles.reserve(*n_particles);
			let init_mean = ifunc(*n_variables);
	
			for _ in 0..*n_particles
			{
				let init = gen_inner_region(&init_mean, sigma);
				let value = ofunc(&init);
				let search_point;
				if value.is_finite()
				{
					search_point = SearchPoint::gen(init, Some(value));
				}
				else
				{
					search_point = SearchPoint::gen(init, None);
				}
				particles.push(Particle::gen(search_point.clone(), vec![0.0_f64 ;*n_variables], search_point));
			}


			let _ = mem::replace(st_particles, Some(particles));	
		}
		{// gbest initialize
			let gbest = self.swarm_best_mut();
			if gbest.is_some()
			{
				*gbest = None;
			}
		}
		{// weight step calculation
			let max_iter = self.max_iter();
			let (w0, wt) = self.wrange();
			*self.step_w_mut() = (wt - w0) / max_iter as f64;
		}
	}

    fn optimeze(&mut self) -> PsoResult
	{
		use std::mem;
		use std::time;
		use rayon::prelude::*;

		let start_time = time::Instant::now();

        self.initialize();
        self.update_swarm();
		{// history record
			if let Some(best) = &self.gbest
			{
				if let Some(value) = best.fvalue()
				{
					self.update_value.push(*value);
				}
				else
				{
					self.update_value.push(f64::NAN);
				}
				self.update_vals.push(best.x().to_vec().clone());
			}
			else
			{
				self.update_value.push(f64::NAN);
				self.update_vals.push(iter::repeat(f64::NAN).take(self.n_variables).collect());
			}

			let particles;
        	{
            	let st_particles = self.particles_mut();
            	particles = mem::replace(st_particles, None).unwrap();
        	}
			let svals =
				particles
					.par_iter()
					.map(|p| p.current().x().to_vec().clone() )
					.collect();
			self.search_vals.push(svals);
			{
            	let st_particles = self.particles_mut();
            	let _ = mem::replace(st_particles, Some(particles));
        	}
		}
//        println!("init : {:?}", self.gbest);
    
	    for _ in 0..self.max_iter
   		{
			self.current_iter += 1;
        	let mut particles;
        	{
            	let st_particles = self.particles_mut();
            	particles = mem::replace(st_particles, None).unwrap();
        	}
        	let gbest;
        	{
           		gbest = mem::replace(&mut self.gbest, None).unwrap();
        	}
	        {// particles moving
	            let gbest_x = gbest.x();

           		particles.par_iter_mut()
                	.for_each(
                   		|p|
                    	{
                        	self.particle_transfer(p, gbest_x);
							if !check_region(&p.current(), self.sigma)
							{
								let init = gen_inner_region(&gbest_x, self.sigma);
								let value = self.ofunc().as_ref().unwrap().clone()(&init);
								let search_point;
								if value.is_finite()
								{
									search_point = SearchPoint::gen(init, Some(value));
								}
								else
								{
									search_point = SearchPoint::gen(init, None);
								}
								*p.current_mut() = search_point;
							}
							p.update();
                    	});
				// record current search point
				if self.current_iter == 1 || self.current_iter % 1000 == 0
				{
					let svals =
						particles
							.par_iter()
							.map(|p| p.current().x().to_vec().clone() )
							.collect();
					self.search_vals.push(svals);
				}
        	}
        	{
            	let st_particles = self.particles_mut();
            	let _ = mem::replace(st_particles, Some(particles));
        	}
        	{
            	let _ = mem::replace(&mut self.gbest, Some(gbest));
        	}
			// gbest(swarm best) update
        	self.update_swarm();
			{// history record
				if let Some(best) = &self.gbest
				{
					if let Some(value) = best.fvalue()
					{
						self.update_value.push(*value);
					}
					else
					{
						self.update_value.push(f64::NAN);
					}
					self.update_vals.push(best.x().to_vec().clone());
				}
				else
				{
					self.update_value.push(f64::NAN);
					self.update_vals.push(iter::repeat(f64::NAN).take(self.n_variables).collect());
				}
			}
    	}

		PsoResult
		{
			vals: self.gbest.as_ref().unwrap().x().to_vec(),
			value: self.gbest.as_ref().unwrap().fvalue().unwrap_or_else(|| f64::NAN),
			update_fvalue: self.update_value.clone(),
			update_evals: 
				(0..self.max_iter).into_par_iter()
					.map(|i| i*self.n_particles as u64)
					.collect::<Vec<u64>>(),
			search_vals: self.search_vals.clone(),
			update_best_vals: self.update_vals.clone(),
			time: start_time.elapsed(),
			evals: self.max_evaluations,
		}
    }

    fn ofunc(&self) -> &Option<Arc<O>>
	{ &self.func }
    fn particles(&self) -> &Option<Vec<Particle>>
	{ &self.particles }
	fn particles_mut(&mut self) -> &mut Option<Vec<Particle>>
	{ &mut self.particles }
    fn particles_vec(&mut self) -> &mut Option<Vec<Particle>>
	{ &mut self.particles }
    fn swarm_best(&self) -> &Option<SearchPoint>
	{ &self.gbest }
    fn swarm_best_mut(&mut self) -> &mut Option<SearchPoint>
	{ &mut self.gbest }
    fn for_update_swarm(&mut self) -> (&mut Option<Vec<Particle>>, &mut Option<SearchPoint>)
	{ (&mut self.particles, &mut self.gbest) }
    fn for_transfer(&mut self) -> (&mut Option<Vec<Particle>>, &Option<SearchPoint>)
	{ (&mut self.particles, &self.gbest) }
    fn n_swarms(&self) -> usize
	{ self.n_swarms }
    fn n_particles(&self) -> usize
	{ self.n_particles }
    fn n_variables(&self) -> usize
	{ self.n_variables }
    fn wrange(&self) -> (f64, f64)
	{ (self.w0, self.wt) }
	fn w(&self) -> f64
	{ self.w }
	fn w_mut(&mut self) -> &mut f64
	{ &mut self.w }
    fn step_w(&self) -> f64
	{ self.step_w }
    fn step_w_mut(&mut self) -> &mut f64
	{ &mut self.step_w }
    fn c(&self) -> (f64, f64)
	{ (self.c1, self.c2) }
    fn max_iter(&self) -> u64
	{ self.max_iter }
    fn current_iter(&self) -> u64
	{ self.current_iter }
    fn current_iter_mut(&mut self) -> &mut u64
	{ &mut self.current_iter }
    fn init_func(&self) -> &Option<Arc<I>>
	{ &self.init_func }
    fn xrange(&self) -> (f64, f64)
	{ (self.lower_x, self.upper_x) }
    fn init_xrange(&self) -> (f64, f64)
	{ (self.init_lx, self.init_ux) }

    fn vrange(&self) -> (f64, f64)
	{ (self.lower_v, self.upper_v) }

    fn init_vrange(&self) -> (f64, f64)
	{ (self.init_lv, self.init_uv) }
}
impl<O, I> Default for LimitedPso<O, I>
where
	O :Fn(&[f64]) -> f64,
	I :Fn(usize) -> Vec<f64>
{
	fn default() -> Self
	{
		Self
		{
			particles :Some(Vec::<Particle>::with_capacity(100)),
			gbest :None,

			n_swarms :1,
			n_particles :100,
			n_variables :0,

			w :0.0,
			step_w: 0.0,
			w0 :0.9,
			wt :0.4,
			c1 :2.0,
			c2 :2.0,

			func :None,
			init_func: None,

			update_value: Vec::<f64>::with_capacity(100000),
			update_evals: Vec::<u64>::with_capacity(100000),
		    update_vals: Vec::<Vec<f64>>::with_capacity(100000),
		    search_vals: Vec::<Vec<Vec<f64>>>::with_capacity(100000),

			max_evaluations :10000000,
		    max_iter: 100000,
		    current_iter: 0,
			max_run :0,

			upper_x :10.0,
			lower_x :-10.0,
			init_ux :10.0,
			init_lx :-10.0,

			upper_v :10.0,
			lower_v :-10.0,
			init_uv :10.0,
			init_lv :-10.0,

			sigma :0.3,
		}
	}
}

pub fn check_region(point :&SearchPoint, limit :f64) -> bool
{
	if point.x().iter().map(|x| x*x ).sum::<f64>() > limit
	{
		return false
	}
	true
}

pub fn gen_inner_region(mean :&[f64], limit :f64) -> Vec<f64>
{

	let mut rng = thread_rng();
	let dist = StandardNormal;

	let adjust_norm = rng.gen::<f64>().powf( (mean.len() as f64).recip() );

	let mut pre_pos = 
		dist.sample_iter(rng)
			.take(mean.len())
			.zip(mean.iter())
			.map(|(x, m) :(f64, &f64)| x + *m)
			.collect::<Vec<f64>>();

	let norm = pre_pos.iter().map(|x| x*x ).sum::<f64>();
	if norm == 0.0
	{
		pre_pos.iter_mut().for_each(|x| *x *= adjust_norm );
		pre_pos
	}
	else
	{
		pre_pos.iter_mut().for_each(|x| *x *= adjust_norm / norm );
		pre_pos
	}
}

