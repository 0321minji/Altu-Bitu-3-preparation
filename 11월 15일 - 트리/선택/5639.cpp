#include<iostream>
#include<vector>

using namespace std;
/*
���� ��ȸ�� �Էµ� ��� ���� ���� ������ ���
���� ��ȸ (�ڱ� �ڽ�->left->right)�� ��������� �ݺ�
���� ��ȸ (left->right->�ڱ� �ڽ�)�� ��������� �ݺ�
 temp�� ����
 temp�� �������� ���� ���� ������ �ش� ���Ҹ� �������� ����,������ ������ �ٽ� postOrder����
*/
void postOrder(int start, int end, vector<int> &tree) {
	if (start > end) {
		return;
	}
	int temp = start + 1;

	for (int i = start + 1; i < end + 1; i++) {
		if (tree[start] < tree[i]) {
			temp = i;
			break;
		}
	}
	postOrder(start + 1, temp-1,tree);
	postOrder(temp, end, tree);
	
	cout << tree[start] << '\n';
}
int main() {
	int input;
	vector<int> tree;
	//tree�� ���� �Է�(������ȸ)
	while (cin >> input) {
		tree.push_back(input);
	}
	//cout << tree.size()<<"\n";
	postOrder(0, tree.size()-1,tree);
}