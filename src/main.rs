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
    init_vel :Vec<f64>,
    init_pos :Vec<f64>,
    pbest_f :f64,
    pbest_pos :Vec<f64>,
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
    fn particle_new(n_number:usize) 
    {
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
            let rand: f64 = rng.gen();
            vel.push(0.0);
            pos.push(0.0);
            init_vel.push(vec![rand ; n_number]);
            init_pos.push(vec![rand ; n_number]);
            pbest_pos.push(0.0);
        }
    }
    
    fn step_opt(n_number:usize)
    {
        let mut rng = rand::thread_rng(); 
        let rand: i32 = rng.gen();

        for i in 0..n_number
        {
            vel[i + 1] = interia * vel[i] + acc1_coef * rand * (pbest_pos[i] - pos[i]) + acc2_coef * rand *(gbest_pos[i] - pos[i]);
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
/// * `N` - 次元
struct PsoParam
{
    t_max :i32,
    n_number :usize,
    inertia :f64,
    acc1_coef :f64,
    acc2_coef :f64,
    N : i32,
}

fn main() {
    println!("Hello, world!");
}
