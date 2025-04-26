package '5-State-based Behavior-1' {
	private import ScalarValues::*;
	private import ISQ::*;
	private import '3a-Function-based Behavior-1'::*;
	package Definitions {
		part def VehicleA {
			perform action 'provide power': 'Provide Power';
			exhibit state 'vehicle states': 'Vehicle States';
		}
		part def VehicleController {
			exhibit state 'controller states': 'Controller States';
		}
		state def 'Vehicle States';
		state def 'Controller States';	
		action def 'Perform Self Test';
		action def 'Apply Parking Brake';
		action def 'Sense Temperature' { out temp: TemperatureValue; }
		attribute def 'Vehicle Start Signal';
		attribute def 'Vehicle On Signal';
		attribute def 'Vehicle Off Signal';
		attribute def 'Start Signal';
		attribute def 'Off Signal';
		attribute def 'Over Temp';
		attribute def 'Return to Normal';
	}
	package Usages {
		private import Definitions::*;
		action 'perform self test': 'Perform Self Test';
		action 'apply parking brake': 'Apply Parking Brake';
		action 'sense temperature': 'Sense Temperature';
		state 'vehicle states': 'Vehicle States' parallel {
			ref vehicle : VehicleA;
			state 'operational states' {			
				entry action initial {
				}
				transition initial then off;
				state off;
				transition 'off-starting'
					first off
					accept 'Vehicle Start Signal' 
					if vehicle1_c1.'brake pedal depressed'
					do send 'Start Signal'() to vehicle1_c1.vehicleController
					then starting {					
					}	
				state starting;
				transition 'starting-on'
					first starting
					accept 'Vehicle On Signal'
					then on;
				state on {				
					entry 'perform self test';
					do 'provide power';
					exit 'apply parking brake';
				}
				transition 'on-off'
					first on
					accept 'Vehicle Off Signal'
					then off;
			}
			state 'health states' {			
				entry action initial;
				do 'sense temperature' { out temp; 
				}
				transition initial then normal;
				state normal;
				transition 'normal-maintenance'
					first normal
					accept at vehicle1_c1.maintenanceTime
					then maintenance;
				transition 'normal-degraded'
					first normal
					accept when 'sense temperature'.temp > vehicle1_c1.Tmax
					do send 'Over Temp'() to vehicle1_c1.vehicleController 
					then degraded;
				state maintenance;
				transition 'maintenance-normal'
					first maintenance
					accept 'Return to Normal'
					then normal;
				state degraded;
				transition 'degraded-normal'
					first degraded
					accept 'Return to Normal'
					then normal;
			}
		}
		state 'controller states': 'Controller States' parallel {
			state 'operational controller states' {
				entry action initial; 
				transition initial then off;
				state off;
				transition 'off-on'
					first off
					accept 'Start Signal'
					then on;
				state on;
				transition 'on-off'
					first on
					accept 'Off Signal'
					then off;
			}
		}		
		part vehicle1_c1: VehicleA {
			port fuelCmdPort {
				in fuelCmd: FuelCmd;
			}
			attribute 'brake pedal depressed': Boolean;		
			attribute maintenanceTime: Time::DateTime;
			attribute Tmax: TemperatureValue;
			perform 'provide power' :>> VehicleA::'provide power' {			
				in fuelCmd = fuelCmdPort.fuelCmd;
			}
			exhibit 'vehicle states' :>> VehicleA::'vehicle states' {
			}	
			//*
			// The above is semantically equivalent to:
			ref state 'vehicle states' :> Usages::'vehicle states', exhibitedStates
				:>> VehicleA::'vehicle states';		
			// For a composite state performance within the vehicle, replace the above with:
			state 'vehicle states' :>> Usages::'vehicle states', VehicleA::'vehicle states';
			*/
			part vehicleController: VehicleController {
				exhibit 'controller states' :>> VehicleController::'controller states';
			}			
		}
	}
}