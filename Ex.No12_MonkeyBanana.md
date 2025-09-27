# Ex.No: 11  Planning –  Monkey Banana Problem
### DATE: 27/09/2025                                                                           
### REGISTER NUMBER : 212223065001
### AIM: 
To find the sequence of plan for Monkey Banana problem using PDDL Editor.
###  Algorithm:
Step 1:  Start the program <br> 
Step 2 : Create a domain for Monkey Banana Problem. <br> 
Step 3:  Create a domain by specifying predicates. <br> 
Step 4: Specify the actions GOTO, CLIMB, PUSH-BOX, GET-KNIFE, GRAB-BANANAS in Monkey Banana problem.<br>  
Step 5:   Define a problem for Monkey Banana problem.<br> 
Step 6:  Obtain the plan for given problem.<br> 
Step 7: Stop the program.<br> 
### Program:

```
(define (domain monkey)
  (:requirements :strips)
  (:constants 
    monkey box knife bananas glass waterfountain
  )
  (:predicates 
    (location ?x)
    (on-floor)
    (at ?o ?x)         ; any object at location ?x
    (hasknife)
    (onbox ?x)
    (hasbananas)
    (hasglass)
    (haswater)
  )

  ;; movement and climbing
  (:action GO-TO
    :parameters (?from ?to)
    :precondition (and (location ?from) (location ?to) (on-floor) (at monkey ?from))
    :effect (and (at monkey ?to) (not (at monkey ?from)))
  )

  (:action CLIMB
    :parameters (?x)
    :precondition (and (location ?x) (at box ?x) (at monkey ?x) (on-floor))
    :effect (and (onbox ?x) (not (on-floor)))
  )

  (:action PUSH-BOX
    :parameters (?from ?to)
    :precondition (and (location ?from) (location ?to) (at box ?from) (at monkey ?from) (on-floor))
    :effect (and (at monkey ?to) (at box ?to) (not (at monkey ?from)) (not (at box ?from)))
  )

  ;; getting bananas
  (:action GET-KNIFE
    :parameters (?x)
    :precondition (and (location ?x) (at knife ?x) (at monkey ?x))
    :effect (and (hasknife) (not (at knife ?x)))
  )

  (:action GRAB-BANANAS
    :parameters (?x)
    :precondition (and (location ?x) (hasknife) (at bananas ?x) (onbox ?x))
    :effect (hasbananas)
  )

  ;; getting water
  (:action PICKGLASS
    :parameters (?x)
    :precondition (and (location ?x) (at glass ?x) (at monkey ?x))
    :effect (and (hasglass) (not (at glass ?x)))
  )

  (:action GETWATER
    :parameters (?x)
    :precondition (and (location ?x) (hasglass) (at waterfountain ?x) (at monkey ?x) (onbox ?x))
    :effect (haswater)
  )
)

```

### Input 

```
(define (problem pb1)
  (:domain monkey)
  (:objects 
    p1 p2 p3 p4 - location
  )
  (:init 
    (location p1)
    (location p2)
    (location p3)
    (location p4)

    (at monkey p1)
    (on-floor)
    (at box p2)
    (at bananas p3)
    (at knife p4)
  )
  (:goal (and (hasbananas)))
)

```

### Output/Plan:

<img width="624" height="535" alt="image" src="https://github.com/user-attachments/assets/cce552ec-a45c-4f2e-8117-ed12bf4314ad" />

<img width="618" height="556" alt="image" src="https://github.com/user-attachments/assets/febee916-21bb-4b91-81b8-aac034866347" />

<img width="642" height="563" alt="image" src="https://github.com/user-attachments/assets/ea5c1cd8-3c59-4080-9258-505ea8697629" />

<img width="660" height="679" alt="image" src="https://github.com/user-attachments/assets/2f5a1556-a891-4ed3-a91d-98763b693ca9" />

<img width="637" height="523" alt="image" src="https://github.com/user-attachments/assets/51cc876b-db4a-40e1-900a-a1b4dc9494ca" />

<img width="550" height="458" alt="image" src="https://github.com/user-attachments/assets/4c08d4fb-b967-40ea-9b53-6fb6a9a18865" />


### Result:
Thus the plan was found for the initial and goal state of given problem.
