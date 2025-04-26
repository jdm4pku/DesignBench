package '15_02-Basic Value Properties' {
	private import ScalarValues::*;
    attribute def LengthValue :> Real {
	}
    part def Tire {
    	attribute manufacturer: String;
        attribute hubDiameter: LengthValue;
        attribute width: Integer;
    }
    part frenchTire: Tire {
    	attribute :>> manufacturer = "Michelin";
    	attribute :>> hubDiameter = 18.0;
    	attribute :>> width = 245;
    }
}
