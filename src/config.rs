extern crate getopts;
use getopts::Options;
use std::env;

pub struct Config {
    pub use_constant_folding: bool,
    pub use_constant_propagation: bool,
    pub use_norm_domain_refinement: bool,
    pub domain_product_threshold: usize,
    pub native_linear_encoding_terms: usize,
    pub use_direct_encoding: bool,
}

impl Config {
    pub fn default() -> Config {
        Config {
            use_constant_folding: true,
            use_constant_propagation: true,
            use_norm_domain_refinement: true,
            domain_product_threshold: 1000,
            native_linear_encoding_terms: 4,
            use_direct_encoding: true,
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

        config
    }
}
