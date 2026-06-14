//! Pricing utilities


/// Cloud provider pricing
pub struct CloudPricing {
    pub provider: String,
    pub gpu_prices: std::collections::HashMap<String, f64>,
}

impl CloudPricing {
    pub fn aws() -> Self {
        let mut prices = std::collections::HashMap::new();
        prices.insert("A100-SXM".to_string(), 3.06); // p4d.24xlarge
        prices.insert("H100-SXM".to_string(), 4.50); // p5.48xlarge
        prices.insert("V100".to_string(), 2.48); // p3.2xlarge
        
        Self {
            provider: "AWS".to_string(),
            gpu_prices: prices,
        }
    }
    
    pub fn gcp() -> Self {
        let mut prices = std::collections::HashMap::new();
        prices.insert("A100-SXM".to_string(), 2.93);
        prices.insert("H100-SXM".to_string(), 4.20);
        prices.insert("V100".to_string(), 2.48);
        
        Self {
            provider: "GCP".to_string(),
            gpu_prices: prices,
        }
    }
    
    pub fn azure() -> Self {
        let mut prices = std::collections::HashMap::new();
        prices.insert("A100-SXM".to_string(), 3.06);
        prices.insert("H100-SXM".to_string(), 4.50);
        prices.insert("V100".to_string(), 2.70);
        
        Self {
            provider: "Azure".to_string(),
            gpu_prices: prices,
        }
    }
    
    pub fn get_price(&self, gpu_name: &str) -> f64 {
        self.gpu_prices.get(gpu_name).copied().unwrap_or(3.0)
    }
}

/// Calculate carbon footprint
pub fn calculate_carbon_footprint(
    energy_kwh: f64,
    region: &str,
) -> f64 {
    // CO2 per kWh varies by region
    let co2_factor = match region {
        "us-east" | "us-west" => 0.43,
        "eu-west" | "eu-central" => 0.233,
        "asia-east" => 0.50,
        _ => 0.40,
    };
    
    energy_kwh * co2_factor
}

/// Calculate PUE-adjusted energy
pub fn calculate_pue_adjusted_energy(
    base_energy_kwh: f64,
    pue: f64,
) -> f64 {
    base_energy_kwh * pue
}
