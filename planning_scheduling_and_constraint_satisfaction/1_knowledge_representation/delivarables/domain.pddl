( define (domain dock_domain)
	( :requirements :strips :negative-preconditions :disjunctive-preconditions )
	( :predicates
		( truck ?x )
		( platform ?x )
		( container ?x ) 
		( on ?x ?y )
		( adj ?x ?y )
		( free ?x )
		( has_container_below ?x )
	)

	( 
		:action move_truck
		:parameters (?t ?source ?target)
		:precondition ( and ( truck ?t ) ( platform ?source) ( platform ?target) ( or ( adj ?source ?target) (adj ?target ?source ) ) ( free ?target ) ( on ?t ?source))
		:effect ( and ( on ?t ?target) ( free ?source )
		        ( not ( free ?target ) ) ( not ( on ?t ?source ) ) )
	)

	(
		:action exchange_pos_truck
		:parameters ( ?t1 ?t2 ?p1 ?p2 )
		:precondition ( and ( truck ?t1 ) ( truck ?t2 ) ( platform ?p1 ) ( platform ?p2 ) ( on ?t1 ?p1 ) ( on ?t2 ?p2 ) ( or ( adj ?p1 ?p2 ) ( adj ?p2 ?p1 ) ) )
		:effect ( and ( on ?t1 ?p2 ) ( on ?t2 ?p1 )
		        ( not ( on ?t1 ?p1 ) ) ( not ( on ?t2 ?p2 ) ) )
	)

	(
		:action load_container_from_platform
		:parameters ( ?t ?c ?p )
		:precondition ( and ( truck ?t ) ( container ?c ) ( platform ?p ) ( free ?t ) ( free ?c ) ( on ?t ?p ) ( on ?c ?p ) ( not ( has_container_below ?c) ) )
		:effect ( and ( on ?c ?t ) 
		        ( not ( on ?c ?p ) ) ( not ( free ?c ) ) ( not ( free ?t ) ) )
	)

	(
		:action load_container_from_container
		:parameters ( ?t ?c_above ?c_below ?p )
		:precondition ( and ( truck ?t ) ( container ?c_above ) ( container ?c_below ) ( platform ?p ) ( on ?c_above ?c_below ) ( on ?c_above ?p ) ( on ?c_below ?p ) ( free ?t ) ( free ?c_above ) ( on ?t ?p ) ( has_container_below ?c_above ) )
		:effect ( and ( on ?c_above ?t ) ( free ?c_below )
		        ( not ( on ?c_above ?c_below ) ) ( not ( has_container_below ?c_above ) ) ( not ( free ?c_above ) ) ( not ( on ?c_above ?p ) ) )
	)

	(
		:action unload_container_to_platform
		:parameters ( ?t ?c ?p )
		:precondition ( and ( truck ?t ) ( container ?c ) ( platform ?p ) ( on ?c ?t ) ( on ?t ?p ) )
		:effect ( and ( free ?t ) ( on ?c ?p ) ( free ?c )
		        ( not ( on ?c ?t ) ) )
	)

	(
		:action unload_container_to_container
		:parameters ( ?t ?p ?c_s ?c_t )
		:precondition ( and ( truck ?t ) ( platform ?p ) ( container ?c_s ) ( container ?c_t ) ( on ?t ?p ) ( on ?c_t ?p ) ( free ?c_t ) ( on ?c_s ?t ) )
		:effect ( and ( on ?c_s ?c_t ) ( on ?c_s ?p ) ( has_container_below ?c_s ) ( free ?t ) ( free ?c_s )
		        ( not ( on ?c_s ?t ) ) ( not ( free ?c_t ) ) )
	)
)
