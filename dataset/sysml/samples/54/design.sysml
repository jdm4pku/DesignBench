package 'VehicleMassAggregation' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	part def MassedThing {
		attribute simpleMass :> ISQ::mass; 
		attribute totalMass :> ISQ::mass;
	}
	part simpleThing : MassedThing {
		attribute :>> totalMass = simpleMass;
	}
	part compositeThing : MassedThing {
		part subcomponents: MassedThing[*];		
		attribute :>> totalMass =
			simpleMass + sum(subcomponents.totalMass); 
	}
	part def CarPart :> MassedThing {			
		attribute serialNumber: String;
	}
	part car: CarPart :> compositeThing {	
		attribute vin :>> serialNumber;
		part carParts: CarPart[*] :>> subcomponents;
		part engine :> simpleThing, carParts {
			//...
		}
		part transmission :> simpleThing, carParts {
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