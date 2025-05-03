package 'TransportationUseCases' {
	part def Vehicle;
	part def Person;
	part def Environment;
	part def 'Fuel Station';
	use case 'provide transportation' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
		actor environment : Environment;
		objective {
		}
		ref :>> start {
			assert constraint {
			}
		}
		first start;
		then include 'enter vehicle' {
			actor :>> driver = 'provide transportation'::driver;
			actor :>> passengers = 'provide transportation'::passengers;
		}
		then use case 'drive vehicle' {
			include 'add fuel'[0..*] {
				actor :>> fueler = driver;
				ref :>> start {
					doc /* Fuel level < 10% max fuel */
				}
			}
		}
		then include 'exit vehicle' {
			actor :>> driver = 'provide transportation'::driver;
			actor :>> passengers = 'provide transportation'::passengers;
		}
		then done;
		ref :>> done {
			assert constraint {
			}
		}
	}
	use case 'enter vehicle' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
	}
	use case 'exit vehicle' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
	}
	use case 'add fuel' {
		subject vehicle : Vehicle;
		actor fueler : Person;
		actor 'fuel station' : 'Fuel Station';
	}
}