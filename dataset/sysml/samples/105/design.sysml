package 'VehiclePowertrainControlLogic' {
	public import Definitions::*;
	public import Usages::*;
	package Definitions {
		alias Torque for ISQ::TorqueValue;
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
			action 'generate torque': 'Generate Torque'{
				in fuelCmd = 'provide power'::fuelCmd;
			}
			flow 'generate torque'.engineTorque 
			    to 'amplify torque'.engineTorque;
			action 'amplify torque': 'Amplify Torque';
			flow 'amplify torque'.transmissionTorque 
			    to 'transfer torque'.transmissionTorque;
			action 'transfer torque': 'Transfer Torque';
			flow 'transfer torque'.driveshaftTorque 
			    to 'distribute torque'.driveShaftTorque;
			action 'distribute torque': 'Distribute Torque';
			first start;
			then merge continue;	
			then action engineStarted accept engineStart: EngineStart;			
			then action engineStopped accept engineOff: EngineOff;	
			then continue;
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