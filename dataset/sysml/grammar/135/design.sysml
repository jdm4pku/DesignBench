package 'ScientificReferenceConstants' {
    private import MeasurementReferences::*;
    private import SI::*;
    private import RealFunctions::*;
    package 'Mathematical Constants' {
        attribute e: Real {
        	assert constraint { round(e * 1E20) == 271828182845904523536.0 }
        }
        attribute pi: Real {
        	assert constraint { round(pi * 1E20) == 314159265358979323846.0 }
        }
    }
    package 'Fundamental Physical Constants' {    
        attribute 'fine structure constant'      : DimensionOneValue = 7.2973525693E-3[one];  
        attribute 'electron to proton mass ratio': DimensionOneValue = 5.44617021487E-4[one]; 
        attribute 'speed of light in vacuum'     : SpeedValue = 299792458[m/s];    
    }
    package 'Global Context' {
        attribute 'nominal earth gravitational acceleration': AccelerationValue = 9.80665['m/s²'];
    }
    package 'Model X Context' {
        attribute 'amplifier gain': DimensionOneValue = 3.5[one];
    }
}
