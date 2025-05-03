package 'VehicleMassAggregation' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	part def MassedThing {
		attribute simpleMass :> ISQ::mass; 
		attribute totalMass :> ISQ::mass default simpleMass;
	}
	part compositeThing : MassedThing {
		part subcomponents: MassedThing[*];		
		attribute :>> totalMass default
			simpleMass + sum(subcomponents.totalMass); 
	}
	part filteredMassThing :> compositeThing {
		attribute minMass :> ISQ::mass;		
		attribute :>> totalMass =
			simpleMass + sum(subcomponents.totalMass.?{in p:>ISQ::mass; p >= minMass});
	}
	part def CarPart :> MassedThing {			
		attribute serialNumber: String;
	}
	part car: CarPart :> compositeThing {	
		attribute vin :>> serialNumber;
		part carParts: CarPart[*] :>> subcomponents;
		part engine :> carParts {
			//...
		}
		part transmission :> carParts {
			//...
		}
	}
	private import SI::kg;
	part c :> car {
		attribute :>> simpleMass = 1000[kg];
		part :>> engine {
			attribute :>> simpleMass = 100[kg];
		}
		part redefines transmission {
			attribute :>> simpleMass = 50[kg];
		}	
	}
	// c::totalMass --> 1150.0[kg]
}