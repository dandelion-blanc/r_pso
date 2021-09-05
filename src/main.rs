use rand::Rng;

fn sphere(x :&[f64])-> f64
{
    1.0
}


/// # 個体
/// # field
/// * `feval` - 現在の評価値
/// * `vel` - 現在の速度
/// * `pos` - 現在の位置
/// * `init_vel` - 初期速度
/// * `init_pos` - 初期位置
/// * `pbest_f` - 最良の評価値
/// * `pbest_pos` - 最良の位置
struct Particle
{
    feval :f64,
    vel :Vec<f64>,
    pos :Vec<f64>,
    pbest_f :f64,
    pbest_pos :Vec<f64>,
}
impl Particle
{
    fn new(n_number:usize)// -> Self
    {
        let feval = 0.0_f64;
        let mut vel = Vec::<f64>::with_capacity(n_number);
        let mut pos = Vec::<f64>::with_capacity(n_number);
        let mut init_vel = Vec::<f64>::with_capacity(n_number);
        let mut init_pos = Vec::<f64>::with_capacity(n_number);
        let mut pbest_f = 0.0_f64;
//        let pbest_pos = Vec::new();

        let mut rng = rand::thread_rng(); 
        for _ in 0..n_number  
        {
            vel.push(0.0);
            pos.push(0.0);
            init_vel.push(rng.gen::<f64>());
            init_pos.push(rng.gen::<f64>());
        }
/*
        Self
        {
            feval,
            vel,
            pos,
            pbest_f,
            pbest_pos,
        }*/
    }
    
    fn ref_mut(&mut self) -> (&mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>)
    {
        (&mut self.pos, &mut self.vel, &mut self.pbest_pos)
    }

    fn step_pos(&mut self) 
    {
        let mut pos = Vec::<f64>::new();
        let vel = Vec::<f64>::new();

        let pv :Vec<f64> = 
            pos.iter()
            .zip(vel)
            .map(|(p, v)| p + v)
            .collect();

    }

    fn step_pbest(&mut self) 
    {
        let pbest_x = Vec::<f64>::new();
        let current_x = Vec::<f64>::new();
        let (current_x, _, pbest_x) = self.ref_mut();

        for i in self.PsoParam.n_particle 
        {
            let score = current_x.iter_mut().sphere();

            if score < pbest_x
            {
                pbest_x = score;
                pbest_x = current_x;   
            }
        }
    }
}

/// # 群れ
/// # field
/// * `particles` - 個体群：
/// * `t` - 更新回数
/// * `gbest_f` - 最良の評価値
/// * `gbest_pos` - 最良の位置
struct Pso
{
    particles :Vec<Particle>,
    t :i32,
    gbest_f :f64,
    gbest_pos :Option<Vec<f64>>,
}
impl Pso
{

    fn step_opt(&mut self, param :&PsoParam)
    {
        let mut rng = rand::thread_rng();
        let w = &param.w;
        let c1 = &param.c1;
        let c2 = &param.c2;
        let gbest = std::mem::replace(&mut self.gbest_pos, None).unwrap();

        self.particles.iter_mut().for_each(
            |p|
            {
                let (current_x, v, pbest_x) = p.ref_mut();

                current_x.iter()
                    .zip(v.iter_mut())
                    .zip(pbest_x.iter())
                    .zip(gbest.iter())
                    .for_each(| ( ( (x, v), px), gx) |
                    	*v = w**v +
                        	rng.gen::<f64>()*c1*(*px - *x) +
                            rng.gen::<f64>()*c2*(*gx - *x) )
            }
        );
        let _ = std::mem::replace(&mut self.gbest_pos, Some(gbest));

        /*
        for i in 0..n_number
        {
            vel[i + 1] = interia * vel[i] + acc1_coef * rng.gen::<f64>() * (pbest_pos[i] - pos[i]) + acc2_coef * rand *(gbest_pos[i] - pos[i]);
            pos[i + 1] = pos[i] + vel[i + 1];

        }

        for i in 0..n_number 
        {
            feval = sphere(pos[i]);

            if pbest_pos[i] > feval 
            {
                for n in 0..N 
                {
                    pbest_pos[i + i * n] = pos[i + i * n];
                }
                pbest_pos[i] = feval;
            }

            if gbest_f > feval 
            {
                gbest_f = feval;
                for n in 0..N 
                {
                    gbest_f[n] = pos[i + i * n];                    
                }    
            }
        }*/
    }

    fn opt()
    {//全体の更新
    
    }
    
}
/// # パラメータ
/// # field
/// * `m_iter` - 最大更新回数
/// * `n_particle` - 個体数
/// * `w` -慣性重み
/// * `c1` - 加速係数
/// * `c2` - 加速係数
/// * `dim` - 次元
struct PsoParam
{
    m_iter :u32,
    n_particle :usize,
    w :f64,
    c1 :f64,
    c2 :f64,
    dim :u32,
}

fn main() {
    println!("Hello, world!");
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let mut rng = rand::thread_rng();
        let mut count = 0_usize;
        let rand :i32 = rng.gen();

        println!("({ })bind value = { }", count, rand);
        count += 1;
        println!("({ })bind value = { }", count, rand);
        count += 1;
        println!("({ })bind value = { }", count, rand);
        count += 1;

        println!("({ })random value = { }", count, rng.gen::<f64>());
        count += 1;
        println!("({ })random value = { }", count, rng.gen::<f64>());
        count += 1;
        println!("({ })random value = { }", count, rng.gen::<f64>());
        count += 1;
    }

    #[test]
    fn test_for_each() {
        let mut vec = vec![1, 2, 3, 4, 5];
        println!("{:?}", vec);

        vec.iter_mut().for_each(|x| *x *= 2);
        println!("{:?}", vec);

        vec.iter().for_each(|x| println!("{ }", x));
    }

    #[test]
    fn test_iter_zip() {
        let a1 = [1, 2, 3];
        let a2 = [4, 5, 6, 7];
        let a3 = [8, 9];

        let mut iter = a1.iter();
        for _ in 0..5
        {
            println!("{:?}", iter.next());
        }
        println!("---------------------------");
        let mut iter = a1.iter();
        while let Some(x) = iter.next()
        {
            println!("{:?}", x);
        }
        println!("---------------------------");

        let mut iter = a1.iter().zip(a2.iter());
        for _ in 0..5
        {
            println!("{:?}", iter.next());
        }
        println!("---------------------------");
        let mut iter = a1.iter().zip(a2.iter());
        while let Some(x) = iter.next()
        {
            println!("{:?}", x);
        }
        println!("---------------------------");

        let mut iter = a1.iter().zip(a2.iter()).zip(a3.iter());
        for _ in 0..5
        {
            println!("{:?}", iter.next());
        }
        println!("---------------------------");
        let mut iter = a1.iter().zip(a2.iter()).zip(a3.iter());
        while let Some(x) = iter.next()
        {
            println!("{:?}", x);
        }
        println!("---------------------------");

    }

    #[test]
    fn test_sum_vec() {
        let mut a1 = vec![1, 2, 3, 4, 5, 7];
        let a2 = vec![4, 5, 6, 7, 8, 9];

        let sum :Vec<_> = 
            a1.iter()
            .zip(a2.iter())
            .map(|(x, y)| x + y)
            .collect()
            ;
        println!("map:{:?}",sum);

        a1.iter_mut()
            .zip(a2.iter())
            .for_each(|(x, y)| *x += *y)
            ;
        println!("for each:{:?}",a1);
    }

    fn test_pbest_pos(&mut self) 
    {
        let pbest_x = vec![3.0, 1.0, 2.0];
        let current_x = vec![1.0, 2.0, 3.0];

        step_pbest();

    }

}
