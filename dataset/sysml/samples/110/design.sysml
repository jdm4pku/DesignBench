package 'VehicleFuelDeliverySystem' {
	private import ScalarValues::Real;
	public import Definitions::*;
	public import Usages::*;
	package Definitions {
		item def Fuel;
		port def FuelPort {
			out item fuel: Fuel;
		}
		part def Pump {
			port fuelInPort : ~FuelPort;
			port fuelOutPort : FuelPort;
		}
		part def StorageTank {
			port fuelOutPort : FuelPort;
		}
		part def FuelTank {
			port fuelInPort : ~FuelPort;
		}
		part def Vehicle {
			port fuelInPort : ~FuelPort;
		}
		action def PumpFuel {
			in fuelIn : Fuel;
			out fuelOut : Fuel;
		}
	}
	package Usages {	
		part context {
			part storageTank : StorageTank;
			flow of  fuel : Fuel
				from storageTank.fuelOutPort.fuel to pump.fuelInPort.fuel {			
			}
			part pump : Pump {
				perform action pumpFuel : PumpFuel {
					in fuelIn = fuelInPort.fuel;
					out fuelOut = fuelOutPort.fuel;
				}
			}
			flow of fuel : Fuel
				from pump.fuelOutPort.fuel to vehicle.fuelInPort.fuel;
			part vehicle : Vehicle {
				flow fuelInPort.fuel to fuelTank.fuel {				
				}
				part fuelTank : FuelTank {
					attribute volumeMax : Real;
					attribute fuelLevel : Real = fuel.volume / volumeMax;	
					item fuel : Fuel {
						attribute volume : Real;
					}
				}
			}
		}
	}
}