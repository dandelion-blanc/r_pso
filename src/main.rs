use rand::Rng;

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
    init_vel :Vec<f64>,
    init_pos :Vec<f64>,
    pbest_f :f64,
    pbest_pos :Vec<f64>,
}
impl particle
{
    fn particle_new() 
    {
        
        let n_number = 100_usize;
        let feval = 0.0_f64;
        let mut vel = Vec::<f64>::with_capacity(n_number);
        let mut pos = Vec::<f64>::with_capacity(n_number);
        let mut init_vel = Vec::<f64>::with_capacity(n_number);
        let mut init_pos = Vec::<f64>::with_capacity(n_number);
        let mut pbest_f = 0.0_f64;
        let mut pbest_pos = Vec::<f64>::with_capacity(n_number);

        for _ in 0..n_number  
        {
            let mut rng = rand::thread_rng(); 
            let rand: i32 = rng.gen();
            vel.push(0.0);
            pos.push(0.0);
            init_vel.push(vec![rand ; n_number]);
            init_pos.push(vec![rand ; n_number]);
            pbest_pos.push(0.0);
        }
        Self
        {
            feval,
            vel,
            pos,
            init_vel,
            init_pos,
            pbest_f,
            pbest_pos,
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
    gbest_pos :Vec<f64>,
}
impl Pso
{
    
    fn step_opt()
    {
        let mut rng = rand::thread_rng(); 
        let rand: i32 = rng.gen();

        for t in 0..t_max {
        vel[t + 1] = interia * vel[t] + acc1_coef * rand * (pbest_pos[t] - pos[t]) + acc2_coef * rand *(gbest_pos[t] - pos[t]);
        pos[t + 1] = pos[t] + vel[t + 1];

        }
    }
    
    fn opt()
    {//全体の更新
    
    }
    
}
/// # パラメータ
/// # field
/// * `t_max` - 最大更新回数
/// * `n_number` - 個体数
/// * `inertia` -慣性重み 
/// * `acc_coef` - 加速係数
struct PsoParam
{
    t_max :i32,
    n_number :usize,
    inertia :f64,
    acc1_coef :f64,
    acc2_coef :f64,
}

fn main() {
    println!("Hello, world!");
}
