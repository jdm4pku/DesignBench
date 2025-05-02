package 'VehicleDynamicsCalculation' {
	private import ScalarValues::Real;
	private import ISQ::*;
	calc def Power { in whlpwr : PowerValue; in Cd : Real; in Cf : Real; in tm : MassValue; in v : SpeedValue;
		attribute drag = Cd * v;
		attribute friction = Cf * tm * v;
		return : PowerValue = whlpwr - drag - friction;
	}
	calc def Acceleration { in tp: PowerValue; in tm : MassValue; in v : SpeedValue;
		return : AccelerationValue = tp / (tm * v);
	}
	calc def Velocity { in dt : TimeValue; in v0 : SpeedValue; in a : AccelerationValue;
		return : SpeedValue = v0 + a * dt;
 	}
	calc def Position { in dt : TimeValue; in x0 : LengthValue; in v : SpeedValue;
		return : LengthValue = x0 + v * dt;
	}
	attribute def DynamicState {
		attribute v: SpeedValue;
		attribute x: LengthValue;
	}
	part def VehicleDynamics {
		attribute C_d : Real;
		attribute C_f : Real;
		attribute wheelPower : PowerValue;
		attribute mass : MassValue;	
		calc updateState { 
			in delta_t : TimeValue; 
			in currState : DynamicState;
			attribute totalPower : PowerValue = Power(wheelPower, C_d, C_f, mass, currState.v);
			return attribute newState : DynamicState {
				:>> v = Velocity(delta_t, currState.v, Acceleration(totalPower, mass, currState.v));
				:>> x = Position(delta_t, currState.x, currState.v);
			}
		}
	} 
}