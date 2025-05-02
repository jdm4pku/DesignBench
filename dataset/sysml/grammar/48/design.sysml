package 'CruiseControlPort' {
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
	part driver_b : Driver {
		port setSpeedPort {
			out setSpeed : SetSpeed;
		}
	}
	interface driverToVehicleInterface connect driver_b.setSpeedPort to vehicle_b.setSpeedPort {
		flow setSpeedFlow of SetSpeed 
			from driver_b.setSpeedPort.setSpeed to vehicle_b.setSpeedPort.setSpeed;
	}
	part vehicle_b : Vehicle {
		port setSpeedPort {
			in setSpeed : SetSpeed;
		}
		bind setSpeedPort = cruiseController_b.setSpeedPort;
		part cruiseController_b : CruiseController {
			port setSpeedPort {
				in setSpeed : SetSpeed;
			}
			port sensedSpeedPort {
				in sensedSpeed : SensedSpeed;
			}
			port fuelCommandPort {
				out fuelCommand : FuelCommand;
			}
		}
		flow sensedSpeedFlow of SensedSpeed 
			from speedometer_b.sensedSpeedPort.sensedSpeed to cruiseController_b.sensedSpeedPort.sensedSpeed;
		part speedometer_b : Speedometer {
			port sensedSpeedPort {
				out sensedSpeed : SensedSpeed;
			}
		}
		flow fuelCommandFlow of FuelCommand 
			from cruiseController_b.fuelCommandPort.fuelCommand to engine_b.fuelCommandPort.fuelCommand;
		part engine_b : Engine {
			port fuelCommandPort {
				in fuelCommand : FuelCommand;
			}
		}
	}
	occurrence cruiseControlInteraction_b : CruiseControlInteraction {
		part :>> driver :>> driver_b {
			port :>> setSpeedPort {
				event driver::setSpeedSent; 
			}
		}
		part :>> vehicle :>> vehicle_b {
			part :>> cruiseController :>> cruiseController_b {
				port :>> setSpeedPort {
					event cruiseController::setSpeedReceived;
				}
			}
			part :>> speedometer :>> speedometer_b {
				port :>> sensedSpeedPort {
					event speedometer::sensedSpeedSent;
				}
			}
			part :>> engine :>> engine_b {
				port :>> fuelCommandPort {
					event engine::fuelCommandReceived;
				}
			}
		}
		message :>> setSpeedMessage = driverToVehicleInterface.setSpeedFlow;
		message :>> sensedSpeedMessage = vehicle_b.sensedSpeedFlow;
		message :>> fuelCommandMessage = vehicle_b.fuelCommandFlow;
	}
}