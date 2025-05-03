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
		ref part :>> driver;		
		ref part :>> vehicle;
		message setSpeedMessage of SetSpeed 
			from driver.setSpeedSent to vehicle.cruiseController.setSpeedReceived;
		message sensedSpeedMessage of SensedSpeed 
			from vehicle.speedometer.sensedSpeedSent to vehicle.cruiseController.sensedSpeedReceived;
		message fuelCommandMessage of FuelCommand 
			from vehicle.cruiseController.fuelCommandSent to vehicle.engine.fuelCommandReceived;
		first setSpeedMessage then sensedSpeedMessage;
	}
}