package 'VehicleSystemConstraint' {
	private import SI::*;
	private import ISQ::*;
	private import NumericalFunctions::*;
	part def Engine;
	part def Transmission;
	constraint def MassConstraint {
		in partMasses : MassValue[0..*];
		in massLimit : MassValue;
		sum(partMasses) <= massLimit
	}
	part def Vehicle {
		constraint massConstraint : MassConstraint {
			in partMasses = (chassisMass, engine.mass, transmission.mass);
			in massLimit = 2500[kg];
		}
		attribute chassisMass : MassValue;
		part engine : Engine {
			attribute mass : MassValue;
		}
		part transmission : Engine {
			attribute mass : MassValue;
		}
	}
	part vehicle1 : Vehicle {
		attribute totalMass : MassValue;			
		assert constraint {totalMass == chassisMass + engine.mass + transmission.mass}	
	}
	part vehicle2 : Vehicle {
		attribute totalMass : MassValue = chassisMass + engine.mass + transmission.mass;
	}
	constraint def Dynamics {
		in mass: MassValue;
		in initialSpeed : SpeedValue;
		in finalSpeed : SpeedValue;
		in deltaT : TimeValue;
		in force : ForceValue;
		force * deltaT == mass * (finalSpeed - initialSpeed) and
		mass > 0[kg]
	}
}