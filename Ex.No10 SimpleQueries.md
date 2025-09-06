# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE: 6/09/2025                                                                            
### REGISTER NUMBER : 212223065001
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
likes(john,X):-food(X).
food(apple).
food(chicken).
eats(sue,X):-eats(bill,X).
eats(bill,peanuts).
```

### Output:

<img width="971" height="209" alt="image" src="https://github.com/user-attachments/assets/4398250a-e2e3-4192-b914-644f4d30a86e" />


### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:

```
likes(steve,X):-easycourse(X).
hard(science).
easycourse(X):-dept(havefun,X).
dept(havefun,bk301).
```

### Output:

<img width="971" height="212" alt="image" src="https://github.com/user-attachments/assets/e9fb1971-8f97-4f18-ac6c-b1315dd36236" />


### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:

```
crime(X):-american(X),weapon(Y),sells(X,Y,Z),hostile(X,Z).
sells(west,Y,nano):-owns(nano,Y),missile(Y).
weapon(Y):-missile(Y).
hostile(B,A):-enemy(A,B).
enemy(nano,west).
owns(nano,m1).
missile(m1).
american(west).
```

### Output:

<img width="956" height="231" alt="image" src="https://github.com/user-attachments/assets/a09ff1c1-9dba-4c21-b519-62ec74a5b888" />


### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
