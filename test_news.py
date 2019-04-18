import sys
import random
import os
from Fact_Checking import fact_extraction, fact_check
from testing_final import testing_tfidf_fe


path = sys.argv[1]
with open(path) as fp:
        string_input=fp.read()
        extracted_data = fact_extraction(string_input)
        flag=1
        return_val1=0
        if(extracted_data!=[]):
                flag=0
                return_val1 = fact_check(extracted_data)

        return_val2=testing_tfidf_fe(string_input)

        if(flag==0):
                if(return_val1==0 or return_val2==0):
                        sys.exit(0)
                else:
                        sys.exit(1)
        else:
                sys.exit(return_val2)


