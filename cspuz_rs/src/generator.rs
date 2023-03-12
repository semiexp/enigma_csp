use rand::seq::SliceRandom;
use rand::Rng;

pub trait Pattern {
    type Output;
    type Update;

    fn initial(&self) -> Self::Output;
    fn enumerate_update_candidates(&self, current: &Self::Output) -> Vec<Self::Update>;
    fn apply_update(&self, current: &Self::Output, update: &Self::Update) -> Self::Output;
}

#[derive(Clone)]
pub struct Choice<T: Clone + PartialEq> {
    candidates: Vec<T>,
    initial: T,
}

impl<T: Clone + PartialEq> Choice<T> {
    pub fn new(candidates: Vec<T>, initial: T) -> Choice<T> {
        Choice {
            candidates,
            initial,
        }
    }
}

impl<T: Clone + PartialEq> Pattern for Choice<T> {
    type Output = T;
    type Update = T;

    fn initial(&self) -> Self::Output {
        self.initial.clone()
    }

    fn enumerate_update_candidates(&self, current: &Self::Output) -> Vec<Self::Update> {
        let mut ret = vec![];
        for cand in &self.candidates {
            if cand != current {
                ret.push(cand.clone());
            }
        }
        ret
    }

    fn apply_update(&self, _current: &Self::Output, update: &Self::Update) -> Self::Output {
        update.clone()
    }
}

impl<T> Pattern for Vec<T>
where
    T: Pattern,
    <T as Pattern>::Output: Clone,
{
    type Output = Vec<<T as Pattern>::Output>;
    type Update = (usize, <T as Pattern>::Update);

    fn initial(&self) -> Self::Output {
        self.iter().map(|x| x.initial()).collect()
    }

    fn enumerate_update_candidates(&self, current: &Self::Output) -> Vec<Self::Update> {
        let mut ret = vec![];
        assert_eq!(self.len(), current.len());
        for i in 0..self.len() {
            for c in self[i].enumerate_update_candidates(&current[i]) {
                ret.push((i, c));
            }
        }
        ret
    }

    fn apply_update(&self, current: &Self::Output, update: &Self::Update) -> Self::Output {
        let (idx, update) = update;
        let mut ret = vec![];
        for i in 0..self.len() {
            if i == *idx {
                ret.push(self[i].apply_update(&current[i], update));
            } else {
                ret.push(current[i].clone());
            }
        }
        ret
    }
}

pub struct Generator<S, P, C, Sc, X, Y>
where
    S: Fn(&X) -> Option<Y>,
    P: Pattern<Output = X>,
    C: Fn(&X, &Y) -> bool,
    Sc: Fn(&X, &Y) -> f64,
{
    solver: S,
    pattern: P,
    checker: C,
    scorer: Sc,
    initial_temperature: f64,
    temperature_decay: f64,
}

enum ProblemAnalysis {
    FullySolved,
    Partial(f64),
    Infeasible,
}

impl<S, P, C, Sc, X, Y> Generator<S, P, C, Sc, X, Y>
where
    S: Fn(&X) -> Option<Y>,
    P: Pattern<Output = X>,
    C: Fn(&X, &Y) -> bool,
    Sc: Fn(&X, &Y) -> f64,
{
    pub fn new(solver: S, pattern: P, checker: C, scorer: Sc) -> Generator<S, P, C, Sc, X, Y> {
        Generator {
            solver,
            pattern,
            checker,
            scorer,
            initial_temperature: 5.0,
            temperature_decay: 0.995,
        }
    }

    pub fn initial_temperature(&mut self, value: f64) -> &mut Generator<S, P, C, Sc, X, Y> {
        self.initial_temperature = value;
        self
    }

    pub fn temperature_decay(&mut self, value: f64) -> &mut Generator<S, P, C, Sc, X, Y> {
        self.temperature_decay = value;
        self
    }

    fn analyze_problem(&self, problem: &X) -> ProblemAnalysis {
        let answer = (self.solver)(problem);
        match answer {
            Some(answer) => {
                if (self.checker)(problem, &answer) {
                    ProblemAnalysis::FullySolved
                } else {
                    let score = (self.scorer)(problem, &answer);
                    ProblemAnalysis::Partial(score)
                }
            }
            None => ProblemAnalysis::Infeasible,
        }
    }

    pub fn generate<R>(&self, rng: &mut R) -> Option<X>
    where
        R: Rng,
    {
        let mut current_problem = self.pattern.initial();
        let mut current_score = -f64::INFINITY;
        let mut temperature = self.initial_temperature;
        let max_steps = 1000;

        for _ in 0..max_steps {
            let mut update_candidates = self.pattern.enumerate_update_candidates(&current_problem);
            update_candidates.shuffle(rng);

            for cand in &update_candidates {
                let updated_problem = self.pattern.apply_update(&current_problem, cand);
                let analysis = self.analyze_problem(&updated_problem);

                match analysis {
                    ProblemAnalysis::Infeasible => continue,
                    ProblemAnalysis::FullySolved => return Some(updated_problem),
                    ProblemAnalysis::Partial(score) => {
                        if current_score < score
                            || rng.gen_range(0.0..1.0)
                                < ((score - current_score) / temperature).exp()
                        {
                            current_problem = updated_problem;
                            current_score = score;
                            break;
                        }
                    }
                }
            }

            temperature *= self.temperature_decay;
        }

        None
    }
}

pub trait DefaultScorableAnswer {
    fn score(&self) -> f64;
    fn fully_solved(&self) -> bool;
}

impl<T> DefaultScorableAnswer for Option<T> {
    fn score(&self) -> f64 {
        if self.is_some() {
            1.0
        } else {
            0.0
        }
    }

    fn fully_solved(&self) -> bool {
        self.is_some()
    }
}

impl<T> DefaultScorableAnswer for Vec<T>
where
    T: DefaultScorableAnswer,
{
    fn score(&self) -> f64 {
        self.iter().map(|x| x.score()).sum()
    }

    fn fully_solved(&self) -> bool {
        self.iter().all(|x| x.fully_solved())
    }
}

impl<T> DefaultScorableAnswer for crate::graph::GridEdges<T>
where
    T: DefaultScorableAnswer,
{
    fn score(&self) -> f64 {
        self.horizontal.score() + self.vertical.score()
    }

    fn fully_solved(&self) -> bool {
        self.horizontal.fully_solved() && self.vertical.fully_solved()
    }
}

impl<T> DefaultScorableAnswer for crate::graph::InnerGridEdges<T>
where
    T: DefaultScorableAnswer,
{
    fn score(&self) -> f64 {
        self.horizontal.score() + self.vertical.score()
    }

    fn fully_solved(&self) -> bool {
        self.horizontal.fully_solved() && self.vertical.fully_solved()
    }
}

pub trait NonDefaultValueCountable<T>
where
    T: PartialEq,
{
    fn count_non_default_value(&self, default: &T) -> i32;
}

impl<T> NonDefaultValueCountable<T> for T
where
    T: PartialEq,
{
    fn count_non_default_value(&self, default: &T) -> i32 {
        if self != default {
            1
        } else {
            0
        }
    }
}

impl<T> NonDefaultValueCountable<T> for Vec<T>
where
    T: PartialEq,
{
    fn count_non_default_value(&self, default: &T) -> i32 {
        self.iter()
            .map(|x| x.count_non_default_value(default))
            .sum()
    }
}

impl<T> NonDefaultValueCountable<T> for Vec<Vec<T>>
where
    T: PartialEq,
{
    fn count_non_default_value(&self, default: &T) -> i32 {
        self.iter()
            .map(|x| x.count_non_default_value(default))
            .sum()
    }
}

pub fn default_uniqueness_checker<A, B>() -> impl Fn(&A, &B) -> bool
where
    B: DefaultScorableAnswer,
{
    |_, answer| answer.fully_solved()
}

pub fn default_scorer<A, B, T>(default: T, clue_weight: f64) -> impl Fn(&A, &B) -> f64
where
    A: NonDefaultValueCountable<T>,
    T: PartialEq,
    B: DefaultScorableAnswer,
{
    move |problem, answer| {
        answer.score() - problem.count_non_default_value(&default) as f64 * clue_weight
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_generator_slitherlink() {
        let height = 5;
        let width = 5;
        let pattern =
            vec![
                vec![Choice::new(vec![None, Some(0), Some(1), Some(2), Some(3)], None); width];
                height
            ];

        let solver = |problem: &Vec<Vec<Option<i32>>>| {
            crate::puzzle::slitherlink::solve_slitherlink(problem)
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let generated = Generator::new(
            solver,
            pattern,
            default_uniqueness_checker(),
            default_scorer(None, 5.0),
        )
        .generate(&mut rng);
        assert!(generated.is_some());
        let generated = generated.unwrap();
        let ans = crate::puzzle::slitherlink::solve_slitherlink(&generated);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        assert!(default_uniqueness_checker()(&generated, &ans));
    }
}
