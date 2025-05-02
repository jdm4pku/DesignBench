package 'CruiseControlSystemRealization' {
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
	part driver_a : Driver {
		action driverBehavior {
			action sendSetSpeed send SetSpeed() to vehicle_a;
		}
	}
	part vehicle_a : Vehicle {
		part cruiseController_a : CruiseController {
			action controllerBehavior {
				action receiveSetSpeed accept SetSpeed via vehicle_a;
				then action receiveSensedSpeed accept SensedSpeed via cruiseController_a;
				then action sendFuelCommand send FuelCommand() to engine_a;
			}
		}	
		part speedometer_a : Speedometer {
			action speedometerBehavior {
				action sendSensedSpeed send SensedSpeed() to cruiseController_a;
			}
		}
		part engine_a : Engine {
			action engineBehavior {
				action receiveFuelCommand accept FuelCommand via engine_a;
			}
		}
	}
	occurrence cruiseControlInteraction_a : CruiseControlInteraction {
		part :>> driver :>> driver_a {
			event driverBehavior.sendSetSpeed[1] :>> setSpeedSent;
		}	
		part :>> vehicle :>> vehicle_a {
			part :>> cruiseController :>> cruiseController_a {
				event controllerBehavior.receiveSetSpeed[1] :>> setSpeedReceived;
				event controllerBehavior.receiveSensedSpeed[1] :>> sensedSpeedReceived;
				event controllerBehavior.sendFuelCommand[1] :>> fuelCommandSent;
			}
			part :>> speedometer :>> speedometer_a {
				event speedometerBehavior.sendSensedSpeed[1] :>> sensedSpeedSent;
			}
			part :>> engine :>> engine_a {
				event engineBehavior.receiveFuelCommand[1] :>> fuelCommandReceived;
			}
		}
		message :>> setSpeedMessage = driver_a.driverBehavior.sendSetSpeed.sentMessage;
		message :>> sensedSpeedMessage = vehicle_a.speedometer_a.speedometerBehavior.sendSensedSpeed.sentMessage;
		message :>> fuelCommandMessage = vehicle_a.cruiseController_a.controllerBehavior.sendFuelCommand.sentMessage;
	}
}