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
/// * `particles` - 
/// * `trytime` - 更新回数
/// * `gbest_f` - 最良の評価値
/// * `gbest_pos` - 最良の位置
struct Pso
{
    particles :Vec<Particle>,
    trytime :u32,
    gbest_f :f64,
    gbest_pos :Vec<f64>,
}
impl Pso
{
    fn step_opt()
    {//一回の更新
    
    }
    
    fn opt()
    {//全体の更新
    
    }
    
}
/// # パラメータ
/// # field
/// * `max_try` - 最大更新回数
/// * `acc_coef` - 加速係数
/// * `n_number` - 個体数
/// * `inertia` -慣性重み 
struct PsoParam
{
    max_try :u32,
    acc_coef :f64,
    n_number :u32,
    inertia :f64,
}

fn main() {
    println!("Hello, world!");
}
