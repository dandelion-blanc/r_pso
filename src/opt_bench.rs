//!
//!
//!
//! # 最適化法用のベンチマーク関数群 <br>
//! 変数列`&[f64]` = `f64`のスライス(`&Vec<f64>`)を受け取り，
//! 評価値として`f64`を返却．
//!
//! ```
//! pub fn function(x :&[f64]) -> f64
//! ```



use crate::itertools::*;
use crate::ord_subset::*;

/// # Sphere Function (Parabolic Function)
/// * 目的関数 ：f(x) = \sum_{i = 1}^{n} x_{i}^{2}
/// * 最適解    ：*x = (0, 0, … ,0)
/// * 最適値    ：f(*x) = 0
pub fn sphere(x :&[f64]) -> f64
{
	x.iter().map(|x| x*x).sum()
}

/// # Rosenbrock's (Saddle) Function
/// * 目的関数 ： f(x) = \sum_{i = 1}^{n - 1} \{ 100(x_{i}^{2} - x_{i + 1})^{2} + (1 - x_{i})^{2} \}
/// * 最適解    ： *x = (1, 1, … ,1)
/// * 最適値    ： f(*x) = 0
pub fn rosenbrock(x :&[f64]) -> f64
{
	x.iter().tuple_windows()
			.map(| (current, next) | 100.0 * ( current.powi(2) - next).powi(2) + (1.0 - current).powi(2))
			.sum()
}

/// # Schwefel's Function(1)
/// * 目的関数 ： f(x) = max{ fabs(x_i) } (1 \leqq i \leqq n) ―＞ (1 <= i <= n)
/// * 最適解    ： *x = (0, 0, … ,0)
/// * 最適値    ： f(*x) = 0
pub fn schwefel1(x :&[f64]) -> f64
{
	*x.iter().ord_subset_max_by_key(|x| x.abs()).unwrap()
}

/// # Branin's Function
/// * 目的関数 ： f(x) = (x_{2} - 5.1*x_{1}^{2} / (4* \pi^{2}) + 5*x_{1} / (\pi) - 6)^{2} + 10*(1 - 1/(8*\pi)cos(x_{1})) + 10
/// * 最適解    ： *x = (-3.142, 12.275), (3.142 ,2.275), (9.425, 2.475)
/// * 最適値    ： f(*x) = 0.398
/// # panic
/// 入力次数が２以外の時にパニック
pub fn branin(x :&[f64]) -> f64
{
	use std::f64::consts;

	assert!(x.len() == 2, "Branin Function can recieve two demension only : input demension = { }", x.len());
	( x[1] - ((5.1*x[0].powi(2)) / (4.0*consts::PI.powi(2))) + ((5.0*x[0]) / consts::PI) - 6.0).powi(2) +
			10.0*(1.0 - 1.0 / (8.0*consts::PI))*f64::cos(x[0]) + 10.0
}

/// # 2^n minima Function
/// * 目的関数 ： f(x) = \sum_{i = 1}^{n} \{ x_{i}^{4} - 16*x_{i}^{2} + 5*x_{i} \}
/// * 最適解    ： *x \approx (-2.90, -2.90, … ,-2.90)
/// * 最適値    ： f(*x) \approx -78n
pub fn minima(x :&[f64]) -> f64
{
	x.iter().map(|x| x.powi(4) - 16.0*x.powi(2) + 5.0*x ).sum()
}

/// # Rastrigin's Function
/// * 目的関数 ： f(x) = \sum_{i = 1}^{n}  \{ x_{i}^{2} - 10*cos(2 \pi x_{i}) + 10 \}
/// * 最適解    ： *x = (0, 0, … ,0)
/// * 最適値    ： f(*x) = 0
pub fn rastrigin(x :&[f64]) -> f64
{
	use std::f64::consts;

	let _y :f64 = x.iter().map(|x| x.powi(2) - 10.0*f64::cos(2.0*consts::PI*x) ).sum();
	_y + 10.0*x.len() as f64
}

/// # Griewank's Function
/// * 目的関数 ： f(x) = (1/4000) \sum_{i = 1}^{n}  \{ x_{i}^{2} \} - \Pi_{i = 1}^{n} \{ cos(x_{i} / i^{0.5}) \} +1
/// * 最適解    ： *x = (0, 0, … ,0)
/// * 最適値    ： f(*x) = 0
pub fn griewank(x :&[f64]) -> f64
{
	let _f1 :f64 = x.iter().map(|x| x.powi(2) ).sum();
	let _f2 :f64 = x.iter().enumerate().map(| (i, x) | f64::cos(x * ((i + 1) as f64).sqrt().recip()) ).product();
	_f1 / 4000.0 - _f2  +1.0
}
