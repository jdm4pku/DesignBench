package 'CruiseControlProtocol' {
	part def Driver;
	part def CruiseController;
	part def Speedometer;
	part def Engine;
	part def Vehicle;
	part driver : Driver {
		event occurrence setSpeedSent;
	}
	part vehicle : Vehicle {
		part cruiseController : CruiseController {
			event occurrence setSpeedReceived;		
			then event occurrence sensedSpeedReceived;		
			then event occurrence fuelCommandSent;
		}
		part speedometer : Speedometer {
			event occurrence sensedSpeedSent;
		}
		part engine : Engine {
			event occurrence fuelCommandReceived;
		}
	}
	item def SetSpeed;
	item def SensedSpeed;
	item def FuelCommand;
	occurrence def CruiseControlInteraction {
		ref part driver : Driver {
			event setSpeedMessage.sourceEvent;
		}
		ref part vehicle : Vehicle {
			part cruiseController : CruiseController {
				event setSpeedMessage.targetEvent;		
				then event sensedSpeedMessage.targetEvent;		
				then event fuelCommandMessage.sourceEvent;
			}
			part speedometer : Speedometer {
				event sensedSpeedMessage.sourceEvent;
			}
			part engine : Engine {
				event fuelCommandMessage.targetEvent;
			}
		}
		message setSpeedMessage of SetSpeed;	
		then message sensedSpeedMessage of SensedSpeed;
		message fuelCommandMessage of FuelCommand;
	}
}