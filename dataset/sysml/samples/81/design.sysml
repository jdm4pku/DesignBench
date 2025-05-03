package 'VehicleConfigurationSpace' {
	private import ScalarValues::Real;
	private import SI::mm;
	attribute def Diameter :> ISQ::LengthValue;
    part def Cylinder {
        attribute diameter : Diameter[1];
    }
    part def Engine {
    	part cylinder : Cylinder[2..*];
    }
    part '4cylEngine' : Engine {
    	part redefines cylinder[4];
    }
    part '6cylEngine' : Engine {
    	part redefines cylinder[6];
    }
	variation attribute def DiameterChoices :> Diameter {
		variant attribute diameterSmall = 70[mm];
		variant attribute diameterLarge = 100[mm];
	}
	variation part def EngineChoices :> Engine {
		variant '4cylEngine';
		variant '6cylEngine';		
	}
	part def Vehicle;
	part def Transmission;
	part manualTransmission;
	part automaticTransmission;
	abstract part vehicleFamily : Vehicle {
		part engine : EngineChoices[1];
		variation part transmission : Transmission[1] {
			variant manualTransmission;
			variant automaticTransmission;
		}
		assert constraint {
			(engine == engine::'4cylEngine' and
			 transmission == transmission::manualTransmission) xor
			(engine == engine::'6cylEngine' and 
			 transmission == transmission::automaticTransmission)
		}	
	}
}