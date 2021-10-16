use crate::gnuplot;
use gnuplot::*;
use crate::serde;


pub trait ResultOptimization
{
	type ResultPrint: std::fmt::Debug + serde::Serialize;
	fn gprint_update_history(&self, filename :&str)
	{
		//use colorlib::HTMLColor_str;

		let mut fg = Figure::new();
		let y = self.update_fvalue();
		let x = self.update_evals()
								.iter()
								.map(|x| *x as f64)
								.collect::<Vec<f64>>();

		fg.set_enhanced_text(true).set_pre_commands("set key font 'Times New Roman, 10.0'")
			.axes2d()
				.set_x_grid(true)
				.set_y_grid(true)
				.set_x_label("Iteration", &[Font("Times New Roman", 12.0)])
				.set_y_label("Objective Function Value", &[Font("Times New Roman", 12.0)])
//				.set_x_range(Fix(0.0), Fix(1.0))
				.set_x_ticks(Some( (Auto, 0) ), &[], &[Font("Times New Roman", 10.0)])
//				.set_y_range(Fix(-1000.0), Fix(1000.0))
				.set_y_ticks(Some( (Auto, 0) ), &[], &[Font("Times New Roman", 10.0)] )
				.lines(&x, y, &[Caption("pso"), Color(&String::from("#000000")), LineWidth(2.0)]);

		fg.save_to_png(filename, 900, 600).expect("gnupolt : gprint update curve is failed.");
	}

	fn tprint_update_history(&self, filename :&str) -> Result<(), Box<dyn std::error::Error>>
	{
		let mut wtr = csv::Writer::from_path(filename)?;

		let mut buf = Vec::with_capacity(4);
		buf.push(format!("fevals"));
		buf.push(format!("fvalue"));
		wtr.write_record(buf)?;

		for (fevals, fvalue) in self.update_evals().iter().zip(self.update_fvalue())
		{
			let mut buf = Vec::with_capacity(4);
			buf.push(format!("{ }", fevals));
			buf.push(format!("{ }", fvalue));
			wtr.write_record(buf)?;
		}

		wtr.flush()?;
		Ok(())
	}

	fn tprint_update_variables(&self, filename :&str) -> Result<(), Box<dyn std::error::Error>>
	{
		let mut wtr = csv::Writer::from_path(filename)?;

		let mut buf = Vec::with_capacity(self.update_vals().len() + 1_usize);
		buf.push(format!("fevals"));
		for i in 0..self.update_vals().get(0).ok_or_else(|| "error: update variables len")?.len()
		{
			buf.push(format!("x[{ }]", i));
		}
		wtr.write_record(buf)?;

		for (i, (evals, vals)) in self.update_evals().iter().zip(self.update_vals()).enumerate()
		{
			let mut buf = Vec::with_capacity(vals.len() + 1_usize);
			buf.push(format!("{ }", evals));
			for variable in vals.iter()
			{
				buf.push(format!("{ }", variable));
			}
			wtr.write_record(buf)?;
		}

		wtr.flush()?;
		Ok(())
	}

    fn update_fvalue(&self) -> &[f64];
	fn update_evals(&self) -> &[u64];
	fn update_vals(&self) -> &[Vec<f64>];
	fn search_vals(&self) -> &[Vec<Vec<f64>>];
    fn best_value(&self) -> f64;
    fn best_vals(&self) -> &[f64];
	fn to_serialize(&self, number: usize) -> Self::ResultPrint;
}

pub trait ResultOptStatistics<T :ResultOptimization>
{
	type StatisticsPrint: std::fmt::Debug + serde::Serialize;

	fn best(&self) -> &T;
	fn worst(&self) -> &T;
	fn to_serialize(&self, number: usize) -> Self::StatisticsPrint;
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PsoResultPrint
{
	no :usize,
	fvalue :f64,
	evals :u64,
	iter :u64,

	time :f64,
//	pub run_state :String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct StatisticsPrint
{
	pub no :usize,											//問題番号
	pub average :f64,										//平均
	pub max :f64,											//最大
	pub min :f64,											//最小
	pub std :f64,												//標準偏差(standard deviationの略)
	pub variance :f64,										//分散

	pub evals_average :f64,								//評価回数の平均
	pub time_average :f64,								//実行時間の平均
}




#[derive(Debug, Clone)]
pub struct PsoStatistics
{
	best :PsoResult,											//最良の最適化結果
	worst :PsoResult,											//最良の最適化結果

	average :f64,															//平均
	max :f64,																//最大
	min :f64,																//最小
	std :f64,																//標準偏差(standard deviationの略)
	variance :f64,														//分散

	ave_time :std::time::Duration,
	ave_evals :f64,
}
impl PsoStatistics
{
	pub fn from_result(result :&[PsoResult]) -> Self
	{
		let samples = result.len();
		let mut statistics = Self::default();
		let mut max = (0_usize, result[0].value);
		let mut min = (0_usize, result[0].value);

		for (i, result) in result.iter().enumerate()
		{
			if result.value > max.1
			{
				max.0 = i;
				max.1 = result.value;
			}
			else if result.value < min.1
			{
				min.0 = i;
				min.1 = result.value;
			}

			let value_a = result.value - statistics.average;
			statistics.average += value_a / (i + 1_usize) as f64;
			let value_b = result.value - statistics.average;
			statistics.variance += value_a*value_b;

			let evals_a = result.evals as f64 - statistics.ave_evals;
			statistics.ave_evals += evals_a as f64 / (i + 1_usize) as f64;

			let time_a = result.time.saturating_sub(statistics.ave_time);
			statistics.ave_time.saturating_add( time_a.checked_div(i as u32 + 1).unwrap_or_else(|| std::time::Duration::MAX) );
		}

		statistics.max = max.1;
		statistics.min = min.1;
		statistics.best = result[min.0].clone();
		statistics.best = result[max.0].clone();
		statistics.variance /= samples as f64;						//Welford's online algorithm
		statistics.std = statistics.variance.sqrt();

		statistics
	}
}
impl ResultOptStatistics<PsoResult> for PsoStatistics
{
    type StatisticsPrint = StatisticsPrint;

	fn best(&self) -> &PsoResult
	{ &self.best }

    fn worst(&self) -> &PsoResult
	{ &self.worst }

    fn to_serialize(&self, number: usize) -> Self::StatisticsPrint
	{
        StatisticsPrint
		{
			no: number,
			average: self.average,
			max: self.max,
			min: self.min,
			std: self.std,
			variance: self.variance,
			evals_average: self.ave_evals,
			time_average: self.ave_time.as_secs_f64(),
		}
	}
}
impl Default for PsoStatistics
{
	fn default() -> Self
	{
		Self
		{
			best :PsoResult::default(),
			worst :PsoResult::default(),

			average :0.0,
			max :0.0,
			min :0.0,
			std :0.0,
			variance :0.0,
			
			ave_time: std::time::Duration::default(),
			ave_evals: 0.0,
		}
	}
}

#[derive(Debug, Clone)]
pub struct PsoResult
{
	pub vals :Vec<f64>,
	pub value :f64,

	pub update_fvalue :Vec<f64>,
	pub update_evals :Vec<u64>,
	pub update_best_vals :Vec<Vec<f64>>,
	pub search_vals :Vec<Vec<Vec<f64>>>,

	pub time :std::time::Duration,
	pub	evals :u64,
}
impl ResultOptimization for PsoResult
{
	type ResultPrint = PsoResultPrint;
    fn update_fvalue(&self) -> &[f64]
	{ &self.update_fvalue }

    fn update_evals(&self) -> &[u64]
	{ &self.update_evals }

    fn update_vals(&self) -> &[Vec<f64>]
	{ &self.update_best_vals }

    fn search_vals(&self) -> &[Vec<Vec<f64>>]
	{ &self.search_vals }

    fn best_value(&self) -> f64
    { self.value }

    fn best_vals(&self) -> &[f64]
    { &self.vals }

    fn to_serialize(&self, number :usize) -> PsoResultPrint
	{
		PsoResultPrint
		{
			no :number,

			fvalue :self.value,
			evals :self.evals,
			iter :*self.update_evals.last().unwrap_or_else(|| &0),

			time :self.time.as_secs_f64(),
//			run_state :source.run_state.to_string(),
		}
    }
}
impl Default for PsoResult
{
    fn default() -> Self
    {
        Self
		{
			vals :Vec::new(),
			value :0.0,

			update_fvalue: Vec::new(),
			update_evals: Vec::new(),
			update_best_vals :Vec::new(),
			search_vals: Vec::new(),
			
			time: std::time::Duration::default(),
			evals: 0,
		}
    }    
}

pub fn tprint_result<R>(result :&[R], dirname :&str) -> Result<(), Box<dyn std::error::Error>>
where
	R: ResultOptimization
{
	let filename = format!("{ }/all_result.csv", dirname);
	let mut wtr = csv::Writer::from_path(filename)?;

	for (i, current_result) in result.iter().enumerate()
	{
		wtr.serialize(current_result.to_serialize(i + 1_usize) )?;
	}

    wtr.flush()?;
    Ok(())
}

pub fn tprint_statistics<S, R>(statistics :&[S], filename :&str) -> Result<(), Box<dyn std::error::Error>>
where
	S: ResultOptStatistics<R>,
	R: ResultOptimization
{
	let mut wtr = csv::Writer::from_path(filename)?;

	for (i, current_statistics) in statistics.iter().enumerate()
	{
		wtr.serialize(current_statistics.to_serialize(i + 1_usize) )?;
	}

    wtr.flush()?;
    Ok(())
}



