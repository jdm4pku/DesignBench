package 'VehiclePowertrainFunctionalArchitecture' {
	public import Definitions::*;
	public import Usages::*;
	package Definitions {
		alias Torque for ISQ::TorqueValue {
		}
		attribute def FuelCmd;
		attribute def EngineStart;
		attribute def EngineOff;
		action def 'Generate Torque' { in fuelCmd: FuelCmd; out engineTorque: Torque; }
		action def 'Amplify Torque' { in engineTorque: Torque; out transmissionTorque: Torque; }
		action def 'Transfer Torque' { in transmissionTorque: Torque; out driveshaftTorque: Torque; }
		action def 'Distribute Torque' { in driveShaftTorque: Torque; out wheelTorque1: Torque; out wheelTorque2: Torque; }
		action def 'Provide Power' { in fuelCmd: FuelCmd; out wheelTorque1: Torque; out wheelTorque2: Torque; }
	}
	package Usages {
		action 'provide power': 'Provide Power'{
			in fuelCmd: FuelCmd;
			out wheelTorque1: Torque; 
			out wheelTorque2: Torque;
			bind 'generate torque'.fuelCmd = fuelCmd {
			}
			action 'generate torque': 'Generate Torque' {
			}
			
			flow 'generate torque'.engineTorque 
			    to 'amplify torque'.engineTorque {
			}
			action 'amplify torque': 'Amplify Torque';
			flow 'amplify torque'.transmissionTorque 
			    to 'transfer torque'.transmissionTorque;
			action 'transfer torque': 'Transfer Torque';
			flow 'transfer torque'.driveshaftTorque 
			    to 'distribute torque'.driveShaftTorque;
			action 'distribute torque': 'Distribute Torque';
			bind wheelTorque1 = 'distribute torque'.wheelTorque1;
			bind wheelTorque2 = 'distribute torque'.wheelTorque2;
			first start then continue {
			}
			merge continue {
			}
			first continue then engineStarted;
			action engineStarted accept engineStart: EngineStart {
			}			
			first engineStarted then engineStopped;
			action engineStopped accept engineOff: EngineOff;	
			first engineStopped then continue;
			first engineStarted then 'generate torque';
			first engineStarted then 'amplify torque';
			first engineStarted then 'transfer torque';
			first engineStarted then 'distribute torque';
			first 'generate torque' then engineStopped;
			first 'amplify torque' then engineStopped;		
			first 'transfer torque' then engineStopped;		
			first 'distribute torque' then engineStopped;		
		}
	}
}