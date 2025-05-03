package 'CompleteTransportationScenario' {
	part def Vehicle;
	part def Person;
	part def Environment;
	part def 'Fuel Station';
	use case def 'Provide Transportation' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
		actor environment : Environment;
		objective {
			doc 
			/* Transport driver and passengers from starting location 
			 * to ending location.
			 */
		}		
	}
	use case def 'Enter Vehicle' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
	}
	use case def 'Exit Vehicle' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
	}
	part def 'Fuel Station';
	use case 'provide transportation' : 'Provide Transportation' {		
		first start;
		then include use case 'enter vehicle' : 'Enter Vehicle' {
			actor :>> driver = 'provide transportation'::driver;
			actor :>> passengers = 'provide transportation'::passengers;
		}
		then use case 'drive vehicle' {
			actor driver = 'provide transportation'::driver;
			actor environment = 'provide transportation'::environment;	
			include 'add fuel'[0..*] { 
				actor :>> fueler = driver;
			}
		}
		then include use case 'exit vehicle' : 'Exit Vehicle' {
			actor :>> driver = 'provide transportation'::driver;
			actor :>> passengers = 'provide transportation'::passengers;
		}
		then done;		
	}
	use case 'add fuel' {
		subject vehicle : Vehicle;
		actor fueler : Person;
		actor 'fuel station' : 'Fuel Station';
	}
}