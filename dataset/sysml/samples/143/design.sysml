package 'CoreDataTypes' {
	private import ScalarValues::Integer {
	}
	private import ScalarValues::Natural;
	attribute def UnsignedInteger :> Natural {	
	}
	private import ScalarValues::Real {
	}
	attribute def UnsignedReal :> Real {
		attribute x: Real :>> self;
		assert constraint { x >= 0.0 }
	}
	private import ScalarValues::String {
	}
	private import ScalarValues::Boolean {
	}
	private import Time::DateTime;
	enum def ConditionColor {	
		enum red;
		enum yellow;
		enum green;
	}
	attribute def ConditionLevel {
		attribute associatedColor : ConditionColor;
	}
	enum def SeverityEnum :> ConditionLevel {
		danger { 
			:>> associatedColor = ConditionColor::red;
		}
		warning { 
			:>> associatedColor = ConditionColor::yellow;
		}
		normal { 
			:>> associatedColor = ConditionColor::green;
		}
	}
	attribute def Diameter :> ISQ::LengthValue;	
	enum def DiameterChoice :> Diameter {
		small = 60 [SI::mm];
		medium = 70 [SI::mm];
		large = 80 [SI::mm];
	}	
	attribute aperatureDiameter: DiameterChoice = DiameterChoice::small;
}