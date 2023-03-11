use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::fold::{self, Fold};
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, parse_quote, BinOp, Expr, Path, Token};

struct Input {
    arg: Expr,
    mod_path: Path,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let arg = input.parse()?;
        input.parse::<Token![,]>()?;
        let mod_path = input.parse()?;
        Ok(Input { arg, mod_path })
    }
}

fn make_explicit_op(lhs: Expr, rhs: Expr, op: Ident, path: &Path) -> Expr {
    parse_quote!(
        #path::solver::ops::#op(&#lhs, &#rhs)
    )
}

fn explicit_op_ident(op: BinOp) -> Option<Ident> {
    let ident_name = match op {
        BinOp::Eq(_) => "eq",
        BinOp::Ne(_) => "ne",
        BinOp::Le(_) => "le",
        BinOp::Lt(_) => "lt",
        BinOp::Ge(_) => "ge",
        BinOp::Gt(_) => "gt",
        _ => return None,
    };
    Some(Ident::new(ident_name, Span::call_site()))
}

struct Converter(Path);

impl Fold for Converter {
    fn fold_expr(&mut self, e: Expr) -> Expr {
        match e {
            Expr::Binary(e) => match explicit_op_ident(e.op) {
                Some(ident) => {
                    let left = fold::fold_expr(self, *e.left);
                    let right = fold::fold_expr(self, *e.right);
                    make_explicit_op(left, right, ident, &self.0)
                }
                _ => Expr::Binary(fold::fold_expr_binary(self, e)),
            },
            Expr::Call(e) if e.args.len() == 2 => {
                let f = fold::fold_expr(self, *e.func);
                let x = fold::fold_expr(self, e.args[0].clone());
                let y = fold::fold_expr(self, e.args[1].clone());
                parse_quote!(
                    crate::solver::ops::call(#f, #x, #y)
                )
            }
            _ => fold::fold_expr(self, e),
        }
    }
}

#[proc_macro]
pub fn _expr_impl(input: TokenStream) -> TokenStream {
    let Input { arg, mod_path } = parse_macro_input!(input as Input);
    let output = Converter(mod_path).fold_expr(arg);

    TokenStream::from(quote!(#output))
}
