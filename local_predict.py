import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import mode
import time, os
import numpy as np
import joblib

start_time = time.time()

# 设置工作路径
os.chdir('D:/OneDrive/代码开发/16 机器学习/pathogen_drug/')

# 对训练数据进行编码
le = LabelEncoder()
train_data = pd.read_csv("trichomonas_valid.csv")
train_data["class"] = le.fit_transform(train_data["class"])

# 加载保存的模型进行预测
best_model = joblib.load("best_model.pkl")

# 计算化合物分子指纹特征和分子级别特征
fp_cols = ['FP'+str(i) for i in range(1024)]
mol_cols = ["NumAtoms", "NumBonds", "NumHeavyAtoms", "NumHeteroatoms", \
            "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings"]

# 利用最优模型对未知化合物进行活性分类，并计算预测准确率score
test_data = pd.read_csv("test.csv", usecols=["compound_id", "canonical_smiles"])
test_data["canonical_smiles"] = test_data["canonical_smiles"].apply(lambda x: Chem.CanonSmiles(x))
test_fps = []
test_mols = []
for smi in test_data["canonical_smiles"]:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=1024)
    t = np.zeros((1,), dtype=bool)
    DataStructs.ConvertToNumpyArray(fp, t) 
    test_fps.append(t)
    num_heteroatoms = 0 if not mol.HasSubstructMatch(Chem.MolFromSmarts("[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]")) else \
            len(mol.GetSubstructMatches(Chem.MolFromSmarts("[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]")))
    test_mols.append([mol.GetNumAtoms(), mol.GetNumBonds(), \
                     mol.GetNumHeavyAtoms(), num_heteroatoms, \
                     AllChem.CalcNumRotatableBonds(mol), \
                     len(Chem.GetSymmSSSR(mol, True)), \
                     len(Chem.GetSymmSSSR(mol, False))])

test_fps = pd.DataFrame(test_fps, columns=fp_cols, dtype=bool)
test_mols = pd.DataFrame(test_mols, columns=mol_cols)
test_features = pd.concat([test_fps, test_mols], axis=1)
#test_data["predict_score"] = best_model.predict_proba(test_features)[:, 1]
test_data["class"] = le.inverse_transform(best_model.predict(test_features))

# 将结果保存在result.csv中
test_data.to_csv("result.csv", columns=["compound_id", "canonical_smiles", "class"], index=False)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time elapsed: {total_time:.2f} seconds")



