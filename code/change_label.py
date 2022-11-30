
import sys
import glob

lastLabel = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=',
               'A_', 'B_', 'C_', 'Delta', 'E_', 'F_', 'G_', 'H_', 'I_', 'L_', 'M_', 'N_', 'P_', 'R_',
               'S_', 'T_', 'V_', 'X_', 'Y_', '[', ']', 'a', 'alpha', 'b', 'beta', 'c', 'cos', 'd', 'div',
               'div_op', 'dot', 'e', 'exists', 'f', 'forall', 'g', 'gamma', 'geq', 'gt', 'h', 'i', 'in',
                    'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'm', 'mu', 'n',
               'neq', 'o', 'p', 'phi', 'pi', 'pipe', 'pm', 'prime', 'q', 'r', 'rightarrow', 's',
               'sigma', 'sin', 'sqrt', 'sum', 't', 'tan', 'theta', 'times', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}']

newLabel = ['!', '(', ')', '+', 'COMMA', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=',
               'A', 'B', 'C', r'\Delta', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R',
               'S', 'T', 'V', 'X', 'Y', '[', ']', 'a', r'\alpha', 'b', r'\beta', 'c', r'\cos', 'd', r'\div',
               '/', '.', 'e', r'\exists', 'f', r'\forall', 'g', r'\gamma', r'\geq', r'\gt', 'h', 'i', r'\in',
                    r'\infty', r'\int', 'j', 'k', 'l', r'\lambda', r'\ldots', r'\leq', r'\lim', r'\log', r'\lt', 'm', r'\mu', 'n',
               r'\neq', 'o', 'p', r'\phi', '\pi', '|', r'\pm', r'\prime', 'q', 'r', r'\rightarrow', 's',
               r'\sigma', r'\sin', r'\sqrt', '\sum', 't', r'\tan', r'\theta', r'\times', 'u', 'v', 'w', 'x', 'y', 'z', r'\{', r'\}']

def replaceAllLabel(text):
    #replace all bad class label
    textUpdated = text

    for i in range(len(newLabel)):
        textUpdated = textUpdated.replace(",{:},".format(lastLabel[i]), r",{:},".format(newLabel[i]))
    
    return textUpdated


def main():
    path = sys.argv[1]
    files = glob.glob("{:}*.lg".format(path))
    for f in files:
        with open(f, 'r') as file :
            filedata = file.read()

        # Replace the target string
        filedata = replaceAllLabel(filedata)

        # Write the file out again
        with open(f, 'w') as file:
            file.write(filedata)

if __name__ == "__main__":
    # execute only if run as a script
    main()