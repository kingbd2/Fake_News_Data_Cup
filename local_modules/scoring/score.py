def calculate_score(TPtrue, FPtrue, TPpart, FPpart, TPfalse, FPfalse, FNtrue, FNpart, FNfalse):
    Ptrue = TPtrue / (TPtrue + FPtrue)
    Ppart = TPpart / (TPpart + FPpart)
    Pfalse = TPfalse / (TPfalse + FPfalse)

    Rtrue = TPtrue / (TPtrue + FNtrue)
    Rpart = TPpart / (TPpart + FNpart)
    Rfalse = TPfalse / (TPfalse + FNfalse)

    P = (Ptrue*Ppart*Pfalse)/3
    R = (Rtrue*Rpart*Rfalse)/3

    print("Precision is: " + str(P))
    print("Recall is: " + str(R))
    
    score = 2*P*R/(P*R)

    return score
