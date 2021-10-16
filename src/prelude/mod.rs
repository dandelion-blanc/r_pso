pub mod result;

use crate::opt_bench;
use result::*;

use std::sync::Arc;
use std::iter;
use std::fmt;
use std::fmt::*;

use crate::rand::*;
use crate::rand::prelude::*;

pub trait Candidate
{
	fn x(&self) -> &[f64];
	fn x_mut(&mut self) -> &mut [f64];
	fn fvalue(&self) -> &Option<f64>;
	fn fvalue_mut(&mut self) -> &mut Option<f64>;
	fn gen(in_x :Vec<f64>, in_fvalue :Option<f64>) -> Self;
}

/// # Searach point struct
/// x :position
/// f :objective function value
///
///
#[derive(Debug, Clone)]
pub struct SearchPoint
{
	x :Vec<f64>,
	fvalue :Option<f64>,
}
impl SearchPoint
{
    fn new(n_variables :&usize) -> Self
	{
		Self
		{
			x :Vec::<f64>::with_capacity(*n_variables),
			fvalue :None,
		}
	}
}
impl Default for SearchPoint
{
	fn default() -> Self
	{
		Self
		{
			x :Vec::new(),
			fvalue :None,
		}
	}	
}
impl Candidate for SearchPoint
{
    fn x(&self) -> &[f64]
	{
        &self.x
    }

    fn x_mut(&mut self) -> &mut [f64]
	{
        &mut self.x
    }

    fn fvalue(&self) -> &Option<f64>
	{
        &self.fvalue
    }

    fn fvalue_mut(&mut self) -> &mut Option<f64>
	{
        &mut self.fvalue
    }

    fn gen(in_x :Vec<f64>, in_fvalue :Option<f64>) -> Self
	{        
		Self
		{
			x :in_x,
			fvalue :in_fvalue,
		}
    }
}
impl fmt::Display for SearchPoint
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		writeln!(f, "SearchPoint")?;
		write!(f, "x : [")?;
		self.x.iter().for_each(|x| write!(f, " {:.4},", x).unwrap());	writeln!(f, "]")?;
		if let Some(value) = self.fvalue
		{
			writeln!(f, "f : {:.4}", value)
		}
		else 
		{
			writeln!(f, "f : None")
		}
	}
}


#[derive(Clone, Debug)]
pub struct Particle
{
	current :SearchPoint,
	velocity :Vec<f64>,

	best :SearchPoint,
}
impl Particle
{
	pub fn new(n_variables :&usize) -> Self
	{
		Self
		{
			current :SearchPoint::new(n_variables),
			velocity :Vec::<f64>::with_capacity(*n_variables),

			best :SearchPoint::new(n_variables),
		}
	}
}
impl Individual<SearchPoint> for Particle
{
    fn current(&self) -> &SearchPoint
	{ &self.current }

    fn current_mut(&mut self) -> &mut SearchPoint
	{ &mut self.current }

    fn best(&self) -> &SearchPoint
	{ &self.best }

    fn best_mut(&mut self) -> &mut SearchPoint
	{ &mut self.best }

    fn velocity(&self) -> &[f64]
	{ &self.velocity }

    fn velocity_mut(&mut self) -> &mut [f64]
	{ &mut self.velocity }

    fn gen(in_current :SearchPoint, in_velocity :Vec<f64>, in_best :SearchPoint) -> Self
	{
		Self
		{
			current :in_current,
			velocity :in_velocity,

			best :in_best,
		}
    }

    fn ref_mut(&mut self) -> (&mut SearchPoint, &mut Vec<f64>, &mut SearchPoint)
	{ (&mut self.current, &mut self.velocity, &mut self.best) }
}
pub trait Individual<T>
where
	T :Candidate + Clone
{
	fn update(&mut self)
	{
		let (current, _, best) = self.ref_mut();
		let current = & *current;
		 
		if best.fvalue().is_none()
		{
			best.clone_from(&current);
		}
		else if current.fvalue().unwrap() < best.fvalue().unwrap()
		{
			best.clone_from(&current);
		}
	}

	fn transfer(current :&mut T, v :&[f64])
	{
//		let (current, v, _) = self.ref_mut();
		let x = current.x_mut();

		x.iter_mut()
			.zip(v)
			.for_each(| (x, v) | *x += *v);
	}

	fn default_init(n_variables :usize) -> Vec<f64>
	{
		let mut rng = thread_rng();
		let range = rand::distributions::Uniform::new(-3.0, 3.0);
		iter::repeat_with(|| range.sample(&mut rng)).take(n_variables).collect::<Vec<f64>>()
	}

	fn current(&self) -> &T;
	fn current_mut(&mut self) -> &mut T;
	fn best(&self) -> &T;
	fn best_mut(&mut self) -> &mut T;
	fn velocity(&self) -> &[f64];
	fn velocity_mut(&mut self) -> &mut [f64];
	fn ref_mut(&mut self) -> (&mut T, &mut Vec<f64>, &mut T);
	fn gen(current :T, velocity :Vec<f64>, best :T) -> Self;
}

pub trait Pso<P, S, R, O, I>
where
	P :Individual<S>,
	S :Candidate + Clone + Default,
	R :ResultOptimization,
	O :Fn(&[f64]) -> f64 + Sync + Send,
	I :Fn(usize) -> Vec<f64> + Sync + Send
{
	fn check_parameters(&self)
	{
		assert!(self.ofunc().is_some(), "Illegal parameter at ofunc(objective function)");
		assert!(self.n_swarms() >= 1_usize, "Illegal parameter at number of swarms");
		assert!(self.n_particles() >= 1_usize, "Illegal parameter at number of particles");
		assert!(self.n_variables() >= 1_usize, "Illegal parameter at number of variables");
		assert!
		(
			self.c().0 > 0.0_f64 &&
			self.c().1 > 0.0_f64,
			"Illegal parameter at pso parameter c1 and c2"
		);
		assert!(self.max_iter() >= 1_u64, "Illegal parameter at max evaluations");
		assert!
		(
			self.xrange().0 < self.xrange().1,
			"Illegal parameter at variable range({ } < x < { })", self.xrange().0, self.xrange().1
		);
		assert!
		(
			self.init_xrange().0 < self.init_xrange().1,
			"Illegal parameter at initilize variable range({ } < x < { })", self.init_xrange().0, self.init_xrange().1
		);
		assert!
		(
			self.vrange().0 < self.vrange().1,
			"Illegal parameter at velocity range({ } < x < { })", self.vrange().0, self.vrange().1
		);
		assert!
		(
			self.init_vrange().0 < self.init_vrange().1,
			"Illegal parameter at initilize velocity range({ } < x < { })", self.init_vrange().0, self.init_vrange().1
		);
	}
	
	fn initialize(&mut self)
	{
		use std::mem;

		self.check_parameters();			// all parameters assertion
		{// particles initialize
			let range = rand::distributions::Uniform::new(self.init_xrange().0, self.init_xrange().1);
			let mut rng = thread_rng();
			let n_variables  = &self.n_variables();
			let n_particles = &self.n_particles();
			let ofunc = self.ofunc().as_ref().unwrap().clone();
			let ifunc = self.init_func().as_ref().unwrap().clone();
			let st_particles = self.particles_vec();
			let mut particles = mem::replace(st_particles, None).unwrap();
				particles.clear();
				particles.reserve(*n_particles);
	
			for _ in 0..*n_particles
			{
//				let init = iter::repeat_with(|| range.sample(&mut rng)).take(*n_variables).collect::<Vec<f64>>();
				let init = ifunc(*n_variables);
				let value = ofunc(&init);
				let search_point;
				if value.is_finite()
				{
					search_point = S::gen(init, Some(value));
				}
				else
				{
					search_point = S::gen(init, None);
				}
				particles.push(P::gen(search_point.clone(), vec![0.0_f64 ;*n_variables], search_point));
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

	fn particle_transfer(&self, particle :&mut P, gbest :&[f64])
	{
		let (lower_x, upper_x) = self.xrange();
		let (lower_v, upper_v) = self.vrange();
		let (c1, c2) = self.c();
		let w = self.w();
		let mut rng = thread_rng();

		// calculate velocity
		{
			let (current, v, pbest) = particle.ref_mut();
			let x = current.x();
			let px = pbest.x();
	
			x.iter()
				.zip(v.iter_mut())
				.zip(px)
				.zip(gbest)
				.for_each(| ( ( (x, v), px), gx) |
					*v = w**v +
						rng.gen::<f64>()*c1*(px - x) +
						rng.gen::<f64>()*c2*(gx - x) );
			for v in v.iter_mut()
			{
				if *v > upper_v
				{
					*v = upper_v;
				}
				else if *v < lower_v
				{
					*v = lower_v;
				}
			}				
		}

		// moving particle
		{
			let (current, v, _) = particle.ref_mut();
			let x = current.x_mut();

			x.iter_mut()
				.zip(v.iter())
				.for_each(| (x, v) | *x += v);
			for x in x.iter_mut()
			{
				if *x > upper_x
				{
					*x = upper_x;
				}
				else if *x < lower_x
				{
					*x = lower_x;
				}
			}
		}

		let func = self.ofunc().as_ref().unwrap().clone();
		let fvalue = func(particle.current().x());
		if fvalue.is_finite()
		{
			*particle.current_mut().fvalue_mut() = Some(fvalue);
		}
		else
		{
			*particle.current_mut().fvalue_mut() = None;
		}
	}

	fn particle_transfer_range_unchecked(&self, particle :&mut P, gbest :&[f64])
	{
		let (c1, c2) = self.c();
		let w = self.w();
		let mut rng = thread_rng();

		// calculate velocity
		{
			let (current, v, pbest) = particle.ref_mut();
			let x = current.x();
			let px = pbest.x();
	
			x.iter()
				.zip(v.iter_mut())
				.zip(px)
				.zip(gbest)
				.for_each(| ( ( (x, v), px), gx) |
					*v = w**v +
						rng.gen::<f64>()*c1*(px - x) +
						rng.gen::<f64>()*c2*(gx - x) );
		}

		// moving particle
		//particle.transfer();
		{
			let (current, v, _) = particle.ref_mut();
			P::transfer(current, v);
		}

		let func = self.ofunc().as_ref().unwrap().clone();
		let fvalue = func(particle.current().x());
		if fvalue.is_finite()
		{
			*particle.current_mut().fvalue_mut() = Some(fvalue);
		}
		else
		{
			*particle.current_mut().fvalue_mut() = None;
		}
	}

	fn update_swarm(&mut self)
	{
		use std::mem;

		{// gbest update
			let (st_particles, st_gbest) = self.for_update_swarm();
			let particles = mem::replace(st_particles, None).unwrap();
			let mut gbest = mem::replace(st_gbest, None).unwrap_or_default();
			let mut min_value = gbest.fvalue().clone();
			let mut min_index = None;
	
			{
				let particles = particles.as_slice();
				for (i, particle) in particles.iter().enumerate()
				{
					if let Some(value) = particle.best().fvalue()
					{
						if min_value.is_none()
						{
							min_index = Some(i);
							min_value = Some(*value);
						}
						else if *value < min_value.unwrap()
						{
							min_index = Some(i);
							min_value = Some(*value);
						}
					}
				}
				if let Some(min_index) = min_index
				{
					gbest.clone_from(particles[min_index].best());
				}
			}
			let _ = mem::replace(st_particles, Some(particles));
			let _ = mem::replace(st_gbest, Some(gbest));	
		}
		{// weight update
			let (w0, _) = self.wrange();
			let step_w = self.step_w();
			let current_iter = self.current_iter();
			*self.w_mut() = w0 + step_w*current_iter as f64;
		}
		{// current iter update
			*self.current_iter_mut() += 1_u64;
		}
	}

	fn optimeze(&mut self) -> R;

	fn ofunc(&self) -> &Option<Arc<O>>;
/*	fn particle(&self) -> &P;
	fn particle_mut(&mut self) -> &mut P;*/
	fn particles(&self) -> &Option<Vec<P>>;
	fn particles_mut(&mut self) -> &mut Option<Vec<P>>;
	fn particles_vec(&mut self) -> &mut Option<Vec<P>>;
	fn swarm_best(&self) -> &Option<S>;
	fn swarm_best_mut(&mut self) -> &mut Option<S>;
	fn for_update_swarm(&mut self) -> (&mut Option<Vec<P>>, &mut Option<S>);
	fn for_transfer(&mut self) -> (&mut Option<Vec<P>>, &Option<S>);
	fn n_swarms(&self) -> usize;
	fn n_particles(&self) -> usize;
	fn n_variables(&self) -> usize;
	fn wrange(&self) -> (f64, f64);
	fn w(&self) -> f64;
	fn w_mut(&mut self) -> &mut f64;
	fn step_w(&self) -> f64;
	fn step_w_mut(&mut self) -> &mut f64;
	fn c(&self) -> (f64, f64);
	fn max_iter(&self) -> u64;
	fn current_iter(&self) -> u64;
	fn current_iter_mut(&mut self) -> &mut u64;
	fn init_func(&self) -> &Option<Arc<I>>;
	fn xrange(&self) -> (f64, f64);
	fn init_xrange(&self) -> (f64, f64);
	fn vrange(&self) -> (f64, f64);
	fn init_vrange(&self) -> (f64, f64);
}

#[derive(Clone)]
pub struct Origin<O, I>
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

}
impl<O, I> Origin<O, I>
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
/*
	pub fn generate(&mut self)
	{
		let n_swarms = self.param.pparam.n_swarms;
		let n_particles = self.param.pparam.n_particles;
		let n_variables = self.param.pparam.n_variables;

		for i in 0..n_swarms
		{
			self.mps.push(ParticleSwarm::new(&n_particles));
			for _ in 0..n_particles
			{
				self.mps[i].particle.push(Particle::new(&n_variables));
			}
		}
	}



	fn step_benchmark(&mut self) -> ResultOptimization
	{
		let cparam = &self.param.cparam;
		let pparam = &self.param.pparam;
		let wstep_size = (pparam.wt - pparam.w0) / cparam.max_t as f64;
		let mut result = ResultOptimization::new();
		result.update_curve.reserve_exact(cparam.max_t);

		for i in 0..cparam.max_t
		{
			self.param.pparam.w = i as f64 *wstep_size + self.param.pparam.w0;

			self.transfer();
			self.update();

			let gbest = &self.mps[self.gmps].gbest;
			result.update_curve.push(self.mps[self.gmps].particle[*gbest].pbest_f);
		}
		result.x = self.mps[self.gmps].particle[self.mps[self.gmps].gbest].pbest_x.clone();
		result.f = self.mps[self.gmps].particle[self.mps[self.gmps].gbest].pbest_f;
		result
	}



	pub fn run(&mut self)
	{
		let mut sum :f64 = 0.0;
		let mut ssum:f64 = 0.0;

		for run in 0..self.param.cparam.max_run
		{
			self.initialize();
			self.step_run();

			let bestp = &self.mps[self.gmps].particle[self.mps[self.gmps].gbest];
			sum += bestp.pbest_f;
			sum += sum * sum;
/*
			if run == 0
			{
				self.resultopt.x = bestp.pbest_x.clone();
				self.resultopt.f = bestp.pbest_f;

				self.statistics.max = bestp.pbest_f;
				self.statistics.min = bestp.pbest_f;
			}
			else
			{
				if bestp.pbest_f < self.resultopt.f
				{
					self.resultopt.x = bestp.pbest_x.clone();
					self.resultopt.f = bestp.pbest_f;
				}
				if bestp.pbest_f > self.statistics.max
				{
					self.statistics.max = bestp.pbest_f;
				}
				if bestp.pbest_f < self.statistics.min
				{
					self.statistics.min = bestp.pbest_f;
				}
			}*/
		}
	}*/
}
impl<O, I> Pso<Particle, SearchPoint, PsoResult, O, I> for Origin<O, I>
where
	O :Fn(&[f64]) -> f64 + Sync + Send,
	I :Fn(usize) -> Vec<f64> + Sync + Send
{
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
				if let Some(value) = best.fvalue
				{
					self.update_value.push(value);
				}
				else
				{
					self.update_value.push(f64::NAN);
				}
				self.update_vals.push(best.x.clone());
			}
			else
			{
				self.update_value.push(f64::NAN);
				self.update_vals.push(iter::repeat(f64::NAN).take(self.n_variables).collect());
			}

			let mut particles;
        	{
            	let st_particles = self.particles_mut();
            	particles = mem::replace(st_particles, None).unwrap();
        	}
			let svals =
				particles
					.par_iter()
					.map(|p| p.current.x.clone() )
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
                        	p.update();
                    	});
				// record current search point
				if self.current_iter == 1 || self.current_iter % 1000 == 0
				{
					let svals =
						particles
							.par_iter()
							.map(|p| p.current.x.clone() )
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
					if let Some(value) = best.fvalue
					{
						self.update_value.push(value);
					}
					else
					{
						self.update_value.push(f64::NAN);
					}
					self.update_vals.push(best.x.clone());
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
impl<O, I> Default for Origin<O, I>
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
		}
	}
}


impl fmt::Display for Particle
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result
	{
		writeln!(f, "Particle");
		writeln!(f, "x : {:?}", self.current.x);
		writeln!(f, "v : {:?}", self.velocity);
		writeln!(f, "pbest_x : {:?}", self.best.x);

		writeln!(f, "f : {:?}", self.current.fvalue);
		writeln!(f, "pbest_f : {:?}", self.best.fvalue)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;
	#[test]
	fn test_init()
	{
		let mut pso = Origin::default()
			.set_particles(&5)
			.set_variables(&10)
//			.set_ofunc_arc(Some(Arc::new(opt_bench::sphere)))		// other definition
			.set_ofunc(opt_bench::sphere)
//			.set_init_func_arc(func)		// other definition
			.set_init_func(Particle::default_init);

		pso.initialize();

		pso.particles.unwrap().iter().for_each(|p| println!("{ }", p));
	}

	#[test]
	fn test_search_point_clone()
	{
		let mut point1 = SearchPoint
		{
			x: vec![0.0, 0.0, 0.0],
			fvalue: Some(0.0_f64),
		};
		let mut point2 = SearchPoint
		{
			x: vec![1.2, 3.3, 4.4],
			fvalue: None,
		};

		point1.clone_from(&point2);
		println!("clone {:?}", point1);
		point2.fvalue = Some(5.0);
		println!("1{:?}", point1);
		println!("2{:?}", point2);
	}
}