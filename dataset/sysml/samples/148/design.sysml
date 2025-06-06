package 'EngineeringMaterialsLibrary' {
	private import ScalarValues::*;
	private import Quantities::*;
	private import MeasurementReferences::*;
	private import SI::*;
    attribute def AtomicMassValue :> MassValue;
	attribute def TensileStrengthUnit :> DerivedUnit {
        private attribute lengthPF: QuantityPowerFactor[1] { :>> quantity = isq.L; :>> exponent = -1; }
        private attribute massPF: QuantityPowerFactor[1] { :>> quantity = isq.M; :>> exponent = 1; }
        private attribute durationPF: QuantityPowerFactor[1] { :>> quantity = isq.T; :>> exponent = -2; }
        attribute :>> quantityDimension { :>> quantityPowerFactors = (lengthPF, massPF, durationPF); }		
	}
    attribute def TensileStrengthValue :> ScalarQuantityValue {
		attribute :>> num: Real;
    	attribute :>> mRef: TensileStrengthUnit;
    }
    attribute <'N/mm²'> 'newton per square millimetre' : TensileStrengthUnit = N / mm^2;
    attribute def Substance;
	attribute def Material :> Substance;
    attribute def Metal :> Material {
        attribute atomicMass: AtomicMassValue[1];
    }
    attribute def Alloy :> Material {
        attribute fractions: MaterialFraction[2..*];
    }
    attribute def MaterialFraction {
        attribute material: Material[1]; 
        attribute massFraction: MassFractionValue[1];
    }
    attribute def MassFractionValue :> DimensionOneValue;    
    attribute Iron: Metal { :>> atomicMass = 55.845[Da]; }
    attribute Carbon: Metal { :>> atomicMass = 12.011[Da]; }
    attribute Manganese: Metal { :>> atomicMass = 54.938[Da]; }
    attribute Steel_980: Alloy {
        private attribute fraction1: MaterialFraction { :>> material = Iron; :>> massFraction = 0.9862[one]; }
        private attribute fraction2: MaterialFraction { :>> material = Carbon; :>> massFraction = 0.0018[one]; }
        private attribute fraction3: MaterialFraction { :>> material = Manganese; :>> massFraction = 0.012[one]; }
    	attribute :>> fractions = (fraction1, fraction2, fraction3);
        attribute tensileStrength: TensileStrengthValue = 980 ['N/mm²'];
    } 
}
