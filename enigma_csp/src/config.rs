extern crate getopts;
use getopts::Options;
use std::env;

#[derive(Clone)]
pub struct Config {
    pub use_constant_folding: bool,
    pub use_constant_propagation: bool,
    pub use_norm_domain_refinement: bool,
    pub domain_product_threshold: usize,
    pub native_linear_encoding_terms: usize,
    pub native_linear_encoding_domain_product_threshold: usize,
    pub use_direct_encoding: bool,
    pub use_log_encoding: bool,
    pub force_use_log_encoding: bool,
    pub direct_encoding_for_binary_vars: bool,
    pub merge_equivalent_variables: bool,
    pub alldifferent_bijection_constraints: bool,
    pub glucose_random_seed: Option<f64>,
    pub glucose_rnd_init_act: bool,
    pub dump_analysis_info: bool,
    pub verbose: bool,
}

impl Config {
    pub const fn default() -> Config {
        Config {
            use_constant_folding: true,
            use_constant_propagation: true,
            use_norm_domain_refinement: true,
            domain_product_threshold: 1000,
            native_linear_encoding_terms: 4,
            native_linear_encoding_domain_product_threshold: 20,
            use_direct_encoding: true,
            use_log_encoding: true,
            force_use_log_encoding: false,
            direct_encoding_for_binary_vars: false,
            merge_equivalent_variables: false,
            alldifferent_bijection_constraints: false,
            glucose_random_seed: None,
            glucose_rnd_init_act: false,
            dump_analysis_info: false,
            verbose: false,
        }
    }

    pub fn parse_from_args() -> Config {
        let args = env::args().collect::<Vec<_>>();
        let mut config = Config::default();
        let mut opts = Options::new();

        let mut bool_flags = [
            (
                &mut config.use_constant_folding,
                "constant-folding",
                "constant folding",
            ),
            (
                &mut config.use_constant_propagation,
                "constant-propagation",
                "constant propagation",
            ),
            (
                &mut config.use_norm_domain_refinement,
                "norm-domain-refinement",
                "domain refinement in normalized CSP",
            ),
            (
                &mut config.use_direct_encoding,
                "direct-encoding",
                "use direct encoding if applicable",
            ),
            (
                &mut config.use_log_encoding,
                "log-encoding",
                "use log encoding if applicable",
            ),
            (
                &mut config.force_use_log_encoding,
                "force-log-encoding",
                "use log encoding for all int variables",
            ),
            (
                &mut config.merge_equivalent_variables,
                "merge-equivalent-variables",
                "merge equivalent variables (which is caused by, for example, (iff x y))",
            ),
            (
                &mut config.alldifferent_bijection_constraints,
                "alldifferent-bijection-constraints",
                "add auxiliary constraints for bijective alldifferent constraints",
            ),
            (
                &mut config.dump_analysis_info,
                "dump-analysis-info",
                "dump analysis info in Glucose",
            ),
            (&mut config.verbose, "verbose", "show verbose outputs"),
        ];
        for (opt, name, desc) in &mut bool_flags {
            if **opt {
                opts.optflag(
                    "",
                    &format!("enable-{}", name),
                    &format!("Enable {} (default).", desc),
                );
                opts.optflag(
                    "",
                    &format!("disable-{}", name),
                    &format!("Disable {}.", desc),
                );
            } else {
                opts.optflag(
                    "",
                    &format!("enable-{}", name),
                    &format!("Enable {}.", desc),
                );
                opts.optflag(
                    "",
                    &format!("disable-{}", name),
                    &format!("Disable {} (default).", desc),
                );
            }
        }
        opts.optopt("", "domain-product-threshold", "Specify the threshold of domain product for introducing an auxiliary variable by Tseitin transformation.", "THRESHOLD");
        opts.optopt("", "native-linear-encoding-terms", "Specify the maximum number of terms in a linear sum which is encoded by the native linear constraint (0 for disabling this).", "TERMS");
        opts.optopt("", "native-linear-encoding-domain-product", "Specify the minimum domain product of linear sums which are encoded by the native linear constraint.", "DOMAIN_PRODUCT");
        opts.optflag("h", "help", "Display this help");

        let matches = match opts.parse(&args[1..]) {
            Ok(m) => m,
            Err(f) => {
                println!("error: {}", f.to_string());
                std::process::exit(1);
            }
        };

        if matches.opt_present("h") {
            // display help
            let brief = format!("Usage: {} [options]", args[0]);
            print!("{}", opts.usage(&brief));
            std::process::exit(0);
        }

        for (opt, name, _) in &mut bool_flags {
            let is_set_enable = matches.opt_present(&format!("enable-{}", name));
            let is_set_disable = matches.opt_present(&format!("disable-{}", name));

            match (is_set_enable, is_set_disable) {
                (true, true) => {
                    println!(
                        "error: conflicting options {} and {} are specified at the same time",
                        name, name
                    );
                    std::process::exit(1);
                }
                (true, false) => **opt = true,
                (false, true) => **opt = false,
                (false, false) => (),
            }
        }

        if let Some(s) = matches.opt_str("domain-product-threshold") {
            let v = match s.parse::<usize>() {
                Ok(v) => v,
                Err(f) => {
                    println!(
                        "error: parse failed for --domain-product-threshold: {}",
                        f.to_string()
                    );
                    std::process::exit(1);
                }
            };
            config.domain_product_threshold = v;
        }
        if let Some(s) = matches.opt_str("native-linear-encoding-terms") {
            let v = match s.parse::<usize>() {
                Ok(v) => v,
                Err(f) => {
                    println!(
                        "error: parse failed for --native-linear-encoding-terms: {}",
                        f.to_string()
                    );
                    std::process::exit(1);
                }
            };
            config.native_linear_encoding_terms = v;
        }
        if let Some(s) = matches.opt_str("native-linear-encoding-domain-product") {
            let v = match s.parse::<usize>() {
                Ok(v) => v,
                Err(f) => {
                    println!(
                        "error: parse failed for --native-linear-encoding-domain-product: {}",
                        f.to_string()
                    );
                    std::process::exit(1);
                }
            };
            config.native_linear_encoding_domain_product_threshold = v;
        }

        config
    }
}
