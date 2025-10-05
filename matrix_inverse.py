import numpy as np

# (1) 행렬 입력 기능
def read_matrix_from_stdin() -> list[list[float]]:
    print("[입력] 정수 n 을 입력하세요 (n×n 정방행렬): ", end="")
    n = int(input().strip())
    if n <= 0:
        raise ValueError("n 은 1 이상이어야 합니다.")
    print(f"[입력] {n}×{n} 행렬의 각 행을 공백으로 구분하여 입력하세요.")

    matrix = []
    for i in range(n):
        row = list(map(float, input(f"{i+1}행: ").split()))
        if len(row) != n:
            raise ValueError("행의 길이가 n 과 같아야 합니다.")
        matrix.append(row)
    return np.array(matrix, dtype=float)


# (2) 행렬식(Determinant)을 이용한 역행렬 계산
def inverse_by_determinant(A: np.ndarray) -> np.ndarray:
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        raise ValueError("역행렬이 존재하지 않습니다 (determinant = 0).")
    return np.linalg.inv(A)


# (3) 가우스–조던 소거법(Gauss–Jordan elimination)을 이용한 역행렬 계산
def inverse_by_gauss_jordan(A: np.ndarray) -> np.ndarray:
    n = len(A)
    # 원본 손상 방지
    A = A.astype(float)
    I = np.eye(n)
    aug = np.hstack((A, I))

    for col in range(n):
        # 부분 피벗팅
        pivot_row = np.argmax(np.abs(aug[col:, col])) + col
        if abs(aug[pivot_row, col]) < 1e-10:
            raise ValueError("역행렬이 존재하지 않습니다 (pivot = 0).")
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # 피벗을 1로 만듦
        aug[col] = aug[col] / aug[col, col]

        # 다른 행을 0으로 만듦
        for row in range(n):
            if row != col:
                aug[row] = aug[row] - aug[row, col] * aug[col]

    return aug[:, n:]


# (4) 결과 비교 및 출력
def compare_matrices(A: np.ndarray, B: np.ndarray, tol=1e-8) -> bool:
    return np.allclose(A, B, atol=tol)


def main():
    try:
        A = read_matrix_from_stdin()
        print("\n입력된 행렬:")
        print(A)

        # 방법1: 행렬식 이용
        inv_det = inverse_by_determinant(A)
        print("\n(1) 행렬식 이용 역행렬:")
        print(inv_det)

        # 방법2: 가우스-조던 이용
        inv_gj = inverse_by_gauss_jordan(A)
        print("\n(2) 가우스-조던 이용 역행렬:")
        print(inv_gj)

        # 비교
        same = compare_matrices(inv_det, inv_gj)
        if same:
            print("\n✅ 두 결과가 동일합니다.")
        else:
            print("\n⚠️ 두 결과가 다릅니다.")

        # (추가기능) A * A^-1 검증
        print("\nA × A⁻¹ =")
        print(np.dot(A, inv_gj))

        # (추가기능) 조건수 출력
        cond = np.linalg.cond(A)
        print(f"\nCondition number: {cond:.4f}")

    except Exception as e:
        print("오류:", e)


if __name__ == "__main__":
    main()
