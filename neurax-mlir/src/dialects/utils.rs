//! Common utilities for NEURAX dialects

use melior::ir::{Attribute, Identifier, Location, Operation, Region, Type, operation::OperationBuilder};
use melior::ir::attribute::FloatAttribute;
use melior::Context;

/// Create a string attribute
pub fn string_attr<'c>(context: &'c Context, value: &str) -> Attribute<'c> {
    Attribute::parse(context, &format!("\"{}\"", value)).unwrap()
}

/// Create an integer attribute
pub fn int_attr<'c>(context: &'c Context, value: i64) -> Attribute<'c> {
    Attribute::parse(context, &format!("{} : i64", value)).unwrap()
}

/// Create a float attribute (F64)
pub fn float_attr<'c>(context: &'c Context, value: f64) -> Attribute<'c> {
    let float_type = Type::float64(context);
    FloatAttribute::new(context, float_type, value).into()
}

/// Create a boolean attribute
pub fn bool_attr<'c>(context: &'c Context, value: bool) -> Attribute<'c> {
    Attribute::parse(context, &format!("{}", value)).unwrap()
}

/// Create an array of integer attributes
pub fn int_array_attr<'c>(context: &'c Context, values: &[i64]) -> Attribute<'c> {
    let elements: Vec<String> = values.iter().map(|v| format!("{} : i64", v)).collect();
    Attribute::parse(context, &format!("[{}]", elements.join(", "))).unwrap()
}

/// Create an identifier
pub fn ident<'c>(context: &'c Context, name: &str) -> Identifier<'c> {
    Identifier::new(context, name)
}

/// Create a simple operation with attributes
pub fn simple_op<'c>(
    context: &'c Context,
    name: &str,
    attrs: &[(&str, Attribute<'c>)],
    location: Location<'c>,
) -> Result<Operation<'c>, melior::Error> {
    let mut builder = OperationBuilder::new(name, location);
    for (key, attr) in attrs {
        builder = builder.add_attributes(&[(Identifier::new(context, key), *attr)]);
    }
    builder.build()
}

/// Create an operation with regions
pub fn op_with_regions<'c>(
    context: &'c Context,
    name: &str,
    attrs: &[(&str, Attribute<'c>)],
    regions: Vec<Region<'c>>,
    location: Location<'c>,
) -> Result<Operation<'c>, melior::Error> {
    let mut builder = OperationBuilder::new(name, location);
    for (key, attr) in attrs {
        builder = builder.add_attributes(&[(Identifier::new(context, key), *attr)]);
    }
    builder = builder.add_regions_vec(regions);
    builder.build()
}
