use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::fold::{self, Fold};
use syn::{parse_macro_input, parse_quote, BinOp, Expr};

fn make_explicit_op(lhs: Expr, rhs: Expr, op: Ident) -> Expr {
    parse_quote!(
        crate::solver::ops::#op(&#lhs, &#rhs)
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

struct Converter;

impl Fold for Converter {
    fn fold_expr(&mut self, e: Expr) -> Expr {
        match e {
            Expr::Binary(e) => match explicit_op_ident(e.op) {
                Some(ident) => {
                    let left = fold::fold_expr(self, *e.left);
                    let right = fold::fold_expr(self, *e.right);
                    make_explicit_op(left, right, ident)
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
pub fn expr(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Expr);
    let output = Converter.fold_expr(input);

    TokenStream::from(quote!(#output))
}
