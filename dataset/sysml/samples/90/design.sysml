library package 'SemanticRiskMetadata' {
	private import ScalarValues::Real;
	private import RiskMetadata::Level;
	abstract occurrence def Situation;
	abstract occurrence situations : Situation[*] nonunique;
	abstract occurrence def Cause {
		attribute probability : Real;
	}
	abstract occurrence causes : Cause[*] nonunique :> situations;
	abstract occurrence def Failure {
		attribute severity : Level;
	}
	abstract occurrence failures : Failure[*] nonunique :> situations;
	abstract connection def Causation :> Occurrences::HappensBefore {
		end cause : Situation[*];
		end effect : Situation[*];
	}
	abstract connection causations : Causation[*] nonunique;
	item def Scenario {
		occurrence :>> situations;
		occurrence :>> causes :> situations;
		occurrence :>> failures :> situations;
	}
	item scenarios : Scenario[*] nonunique;
	private import Metaobjects::SemanticMetadata;
	metadata def situation :> SemanticMetadata {
		:>> baseType = situations meta SysML::Usage;
	}
	metadata def cause :> SemanticMetadata {
		:>> baseType = causes meta SysML::Usage;
	}
	metadata def failure :> SemanticMetadata {
		:>> baseType = failures meta SysML::Usage;
	}
	metadata def causation :> SemanticMetadata {
		:>> baseType = causations meta SysML::Usage;
	}
	metadata def scenario :> SemanticMetadata {
		:>> baseType = scenarios meta SysML::Usage;
	}
}