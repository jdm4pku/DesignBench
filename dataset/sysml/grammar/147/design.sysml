package 'CollectionTypes' {
	private import ScalarValues::*;
	private import Collections::*;
	part def SparePart;
	part def Person;
	attribute def 'Bag<SparePart>' :> Bag {
		ref part :>> elements: SparePart;
	}
	attribute def 'List<Integer>' :> List {
		value :>> elements: Integer;
	}
	attribute def 'Set<String>' :> Set {
		attribute :>> elements: String;
	}
	attribute def 'OrderedSet<Person>' :> OrderedSet {
		ref part :>> elements: Person;
	}
	attribute def 'List<Set<Person>>' :> List {
		attribute :>> elements: Set {
			ref part :>> elements: Person;
		}
	}
	attribute def 'Array<Real>[4]' :> Array {
		attribute :>> elements: Real;
		attribute :>> dimensions = 4;
	}
}