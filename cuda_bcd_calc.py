from numba import cuda
import numpy as np
import cupy
import numba
import math

cpu_c_s10 = np.array(
    [
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
        10000000000,
    ],
    np.int32,
)


@cuda.jit(cache=True)
def gpu_convert_section(byte_list, step_end, n_size, p, precision, has_sign):
    """ GPU上でバイトのセクションを10進数に変換します。 """
    c_s10 = cuda.const.array_like(cpu_c_s10)
    result = 0
    for idx in range(min(n_size, precision - has_sign - step_end)):
        result += (byte_list[p - step_end - idx] - 48) * c_s10[idx]
    return result


@cuda.jit(cache=True)
def gpu_to_dec(byte_list, start_index, precision, scale, result):
    """ GPU上でバイトのリストを10進数に変換します。 """

    # '+' or '-' or ' ' ※+が省略されてスペースになっていることがある
    has_sign = byte_list[start_index] in [43, 45, 32]
    sign = -1 if byte_list[start_index] == 45 else 1  # '-' の場合は符号を負に

    n_size = 9  # int32で安全に扱える桁数
    result[4] = scale
    result[5] = sign
    result[6] = has_sign

    p = start_index + precision - 1
    # 0, 9, 18, 27 の 4ステップ。
    for step in range(4):
        step_end = step * n_size
        if precision > step_end:
            result[step] = gpu_convert_section(
                byte_list, step_end, n_size, p, precision, has_sign
            )
    return


@cuda.jit(cache=True)
def gpu_to_dec_by_col_idx(byte_list, index_start_end, scale, result):
    precision = index_start_end[1] - index_start_end[0]
    return gpu_to_dec(byte_list, index_start_end[0], precision, scale, result)


@cuda.jit(cache=True)
def gpu_set_dec_by_col_idx(byte_list, index_start_end, value):
    gpu_set_dec(byte_list, index_start_end[0], index_start_end[1], value)

@cuda.jit(cache=True)
def gpu_set_dec(byte_list, start, end, value):
    """ decimalをメモリ（ASCIIコード）に書き込む。 """
    c_s10 = cuda.const.array_like(cpu_c_s10)
    length = end - start
    n_step = (length - 1) // 9 + 1
    n_byte = (length - 1) % 9 + 1
    # カウンター式のほうが計算回数は少なく済むような気がする
    counter = end - 1
    for i_step in range(n_step - 1):
        for i_byte in range(9):
            byte_list[counter] = (value[i_step] // c_s10[i_byte]) % 10 + 48
            counter -= 1

    for i_byte in range(n_byte):
        byte_list[counter] = (value[(n_step - 1)] // c_s10[i_byte]) % 10 + 48
        counter -= 1

    # 符号を設定
    if value[6] == 1:
        byte_list[start] = 43 if value[5] == 1 else 45
    else:
        pass  # 符号なしの場合はそのまま


@cuda.jit(cache=True)
def gpu_to_dec_by_int32(value, scale, has_sign, result):
    """ GPU上でint32の値を10進数に変換します。 """
    if value < 0:
        result[0] = -value
        result[5] = -1
    else:
        result[0] = value
        result[5] = 1

    result[1] = 0
    result[2] = 0
    result[3] = 0
    result[4] = scale
    result[6] = has_sign


@cuda.jit(cache=True)
def gpu_convert_int32_to_digit(byte_list, start_end, value):
    c_s10 = cuda.const.array_like(cpu_c_s10)

    """ GPU上でバイトのセクションを10進数に変換します。 """
    for idx in range(start_end[1], -1, -1):
        byte_list[idx] = (value // c_s10[idx]) % 10 + 48
    return


@cuda.jit(cache=True)
def gpu_get_scale_of_block(target):
    """ GPU上で指定されたブロックのスケールを取得します。 """
    block = 0
    for i in range(3, -1, -1):
        if target[i] != 0:
            block = i
            break
    return block


@cuda.jit(cache=True)
def gpu_get_scale_of_number(target, block_index):
    """ GPU上で指定された数値のスケールを取得します。 """
    if target[block_index] <= 0:
        return 0
    else:
        return int(math.log10(target[block_index])) + 1
    # こっちのやり方でも速度はほぼ変わらない。
    if target[block_index] == 0:
        return 0
    elif target[block_index] < 10:
        return 1
    elif target[block_index] < 100:
        return 2
    elif target[block_index] < 1000:
        return 3
    elif target[block_index] < 10000:
        return 4
    elif target[block_index] < 100000:
        return 5
    elif target[block_index] < 1000000:
        return 6
    elif target[block_index] < 10000000:
        return 7
    elif target[block_index] < 100000000:
        return 8
    elif target[block_index] < 1000000000:
        return 9
    else:
        # print("Error: Invalid input. Please enter a valid integer or float.")
        return -1


@cuda.jit(cache=True)
def gpu_set_overflow(target):
    """ GPU上でオーバーフローを設定します。 """
    # print("Error: Overflow")
    target[3] = 999_999_999
    target[2] = 999_999_999
    target[1] = 999_999_999
    target[0] = 999_999_999


@cuda.jit(cache=True)
def gpu_shift_left(target, count):
    """
    指定された配列の要素を指定された桁数だけ左にシフトします。
    左シフトなのでオーバーフローを考慮する。
    """
    c_s10 = cuda.const.array_like(cpu_c_s10)
    if count == 0:
        pass
    elif count <= 9:
        count = count - 0
        count_r = 9 - count
        if count_r - gpu_get_scale_of_number(target, 3) < 0:
            gpu_set_overflow(target)
            return
        target[3] = (target[3] % c_s10[count_r]) * c_s10[count]
        target[3] += target[2] // c_s10[count_r]
        target[2] = (target[2] % c_s10[count_r]) * c_s10[count]
        target[2] += target[1] // c_s10[count_r]
        target[1] = (target[1] % c_s10[count_r]) * c_s10[count]
        target[1] += target[0] // c_s10[count_r]
        target[0] = (target[0] % c_s10[count_r]) * c_s10[count]
    elif count <= 18:
        count = count - 9
        count_r = 9 - count
        if count_r - gpu_get_scale_of_number(target, 2) < 0:
            gpu_set_overflow(target)
            return
        target[3] = (target[2] % c_s10[count_r]) * c_s10[count]
        target[3] += target[1] // c_s10[count_r]
        target[2] = (target[1] % c_s10[count_r]) * c_s10[count]
        target[2] += target[0] // c_s10[count_r]
        target[1] = (target[0] % c_s10[count_r]) * c_s10[count]
        target[0] = 0
    elif count <= 27:
        count = count - 18
        count_r = 9 - count
        if count_r - gpu_get_scale_of_number(target, 1) < 0:
            gpu_set_overflow(target)
            return
        target[3] = (target[1] % c_s10[count_r]) * c_s10[count]
        target[3] += target[0] // c_s10[count_r]
        target[2] = (target[0] % c_s10[count_r]) * c_s10[count]
        target[1] = 0
        target[0] = 0
    elif count <= 36:
        count = count - 27
        count_r = 9 - count
        if count_r - gpu_get_scale_of_number(target, 0) < 0:
            gpu_set_overflow(target)
            return
        target[3] = (target[0] % c_s10[count_r]) * c_s10[count]
        target[2] = 0
        target[1] = 0
        target[0] = 0
    else:
        gpu_set_overflow(target)
        return

@cuda.jit(cache=True)
def gpu_shift_right_round(target, count):
    """
    指定された配列の要素を指定された桁数だけ右にシフトします。
    右シフトなのでオーバーフローは考慮しない代わりに、四捨五入を行う。
    """
    c_s10 = cuda.const.array_like(cpu_c_s10)
    if count == 0:
        return
    count_r = 9 - count
    tail_num = 0
    if count <= 9:
        tail_num = target[0] % c_s10[count] // c_s10[count - 1]
        count_r = 9 - count
        target[0] = target[0] // c_s10[count]
        target[0] += (target[1] % c_s10[count]) * c_s10[count_r]
        target[1] = target[1] // c_s10[count]
        target[1] += (target[2] % c_s10[count]) * c_s10[count_r]
        target[2] = target[2] // c_s10[count]
        target[2] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[3] = target[3] // c_s10[count]
    elif count <= 18:
        tail_num = target[1] % c_s10[count] // c_s10[count - 1]
        count = count - 9
        count_r = 9 - count
        target[0] = target[1] // c_s10[count]
        target[0] += (target[2] % c_s10[count]) * c_s10[count_r]
        target[1] = target[2] // c_s10[count]
        target[1] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[2] = target[3] // c_s10[count]
        target[3] = 0
    elif count <= 27:
        tail_num = target[2] % c_s10[count] // c_s10[count - 1]
        count = count - 18
        count_r = 9 - count
        target[0] = target[2] // c_s10[count]
        target[0] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[1] = target[3] // c_s10[count]
        target[2] = 0
        target[3] = 0
    elif count <= 36:
        tail_num = target[3] % c_s10[count] // c_s10[count - 1]
        count = count - 27
        count_r = 9 - count
        target[0] = target[3] // c_s10[count]
        target[1] = 0
        target[2] = 0
        target[3] = 0
    else:
        tail_num = 0
        target[3] = 0
        target[2] = 0
        target[1] = 0
        target[0] = 0

    # 四捨五入
    if tail_num >= 5:
        target[0] += 1  # 五入


@cuda.jit(cache=True)
def gpu_shift_right_ceil(target, count):
    """
    指定された配列の要素を指定された桁数だけ右にシフトします。
    右シフトなのでオーバーフローは考慮しない代わりに、四捨五入を行う。
    """
    c_s10 = cuda.const.array_like(cpu_c_s10)
    if count == 0:
        return
    count_r = 9 - count
    tail_num = 0
    if count <= 9:
        count_r = 9 - count
        tail_num = (target[0] % c_s10[count]) * c_s10[count_r]
        target[0] = target[0] // c_s10[count]
        target[0] += (target[1] % c_s10[count]) * c_s10[count_r]
        target[1] = target[1] // c_s10[count]
        target[1] += (target[2] % c_s10[count]) * c_s10[count_r]
        target[2] = target[2] // c_s10[count]
        target[2] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[3] = target[3] // c_s10[count]
    elif count <= 18:
        count = count - 9
        count_r = 9 - count
        tail_num = target[0]
        tail_num += (target[1] % c_s10[count]) * c_s10[count_r]
        target[0] = target[1] // c_s10[count]
        target[0] += (target[2] % c_s10[count]) * c_s10[count_r]
        target[1] = target[2] // c_s10[count]
        target[1] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[2] = target[3] // c_s10[count]
        target[3] = 0
    elif count <= 27:
        count = count - 18
        count_r = 9 - count
        tail_num = target[0] + target[1]
        tail_num += (target[2] % c_s10[count]) * c_s10[count_r]
        target[0] = target[2] // c_s10[count]
        target[0] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[1] = target[3] // c_s10[count]
        target[2] = 0
        target[3] = 0
    elif count <= 36:
        count = count - 27
        count_r = 9 - count
        tail_num = target[0] + target[1] + target[2]
        tail_num += (target[3] % c_s10[count]) * c_s10[count_r]
        target[0] = target[3] // c_s10[count]
        target[1] = 0
        target[2] = 0
        target[3] = 0
    else:
        tail_num = 0
        target[3] = 0
        target[2] = 0
        target[1] = 0
        target[0] = 0

    # 切り上げ
    if tail_num > 0:
        target[0] += 1  # 切り上げ


@cuda.jit(cache=True)
def gpu_shift_right_floor(target, count):
    """
    指定された配列の要素を指定された桁数だけ右にシフトします。
    右シフトなのでオーバーフローは考慮しない代わりに、切り捨て。
    """
    c_s10 = cuda.const.array_like(cpu_c_s10)
    if count == 0:
        return
    count_r = 9 - count
    if count <= 9:
        count_r = 9 - count
        target[0] = target[0] // c_s10[count]
        target[0] += (target[1] % c_s10[count]) * c_s10[count_r]
        target[1] = target[1] // c_s10[count]
        target[1] += (target[2] % c_s10[count]) * c_s10[count_r]
        target[2] = target[2] // c_s10[count]
        target[2] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[3] = target[3] // c_s10[count]
    elif count <= 18:
        count = count - 9
        count_r = 9 - count
        target[0] = target[1] // c_s10[count]
        target[0] += (target[2] % c_s10[count]) * c_s10[count_r]
        target[1] = target[2] // c_s10[count]
        target[1] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[2] = target[3] // c_s10[count]
        target[3] = 0
    elif count <= 27:
        count = count - 18
        count_r = 9 - count
        target[0] = target[2] // c_s10[count]
        target[0] += (target[3] % c_s10[count]) * c_s10[count_r]
        target[1] = target[3] // c_s10[count]
        target[2] = 0
        target[3] = 0
    elif count <= 36:
        count = count - 27
        count_r = 9 - count
        target[0] = target[3] // c_s10[count]
        target[1] = 0
        target[2] = 0
        target[3] = 0
    else:
        target[3] = 0
        target[2] = 0
        target[1] = 0
        target[0] = 0


@cuda.jit(cache=True)
def gpu_align_scale_larger(a, b, result_a, result_b):
    """ Aligns the scale of two decimal arrays by scaling the larger one to match the scale of the smaller one. """
    # スケールを合わせる処理を実装
    # a, b: decimal相当配列
    # result_a, result_b: スケールを合わせた結果のdecimal相当配列

    # スケールの取得
    scale_a = a[4]
    scale_b = b[4]

    for i in range(4):
        result_a[i] = a[i]
        result_b[i] = b[i]

    # スケールが大きい方に合わせる
    if scale_a > scale_b:
        # bのスケールをaに合わせる
        gpu_shift_left(result_b, scale_a - scale_b)
        result_a[4] = scale_a
        result_b[4] = scale_a
    elif scale_a < scale_b:
        # aのスケールをbに合わせる
        gpu_shift_left(result_a, scale_b - scale_a)
        result_a[4] = scale_b
        result_b[4] = scale_b
    else:
        pass

    # 符号とメタ情報をコピー
    result_a[5] = a[5]
    result_a[6] = a[6]
    result_b[5] = b[5]
    result_b[6] = b[6]


@cuda.jit(cache=True)
def gpu_align_scale_smaller(a, b, result_a, result_b):
    # スケールを合わせる処理を実装
    # a, b: decimal相当配列
    # result_a, result_b: スケールを合わせた結果のdecimal相当配列

    # スケールの取得
    scale_a = a[4]
    scale_b = b[4]

    for i in range(4):
        result_a[i] = a[i]
        result_b[i] = b[i]

    # スケールが小さい方に合わせる
    if scale_a < scale_b:
        # bのスケールをaに合わせる
        gpu_shift_right_round(result_b, scale_b - scale_a)
        result_a[4] = scale_a
        result_b[4] = scale_a
    elif scale_a > scale_b:
        # aのスケールをbに合わせる
        gpu_shift_right_round(result_a, scale_a - scale_b)
        result_a[4] = scale_b
        result_b[4] = scale_b
    else:
        pass

    # 符号とメタ情報をコピー
    result_a[5] = a[5]
    result_a[6] = a[6]
    result_b[5] = b[5]
    result_b[6] = b[6]


@cuda.jit(cache=True)
def gpu_is_zero(result) -> bool:
    # 結果が 0 の場合は、符号を正に設定
    is_zero = True
    for i in range(4):
        if result[i] != 0:
            is_zero = False
            break
    return is_zero


@cuda.jit(cache=True)
def gpu_abs_compare(a, b):
    # 絶対値で比較
    # a, b: decimal相当配列
    # decimal相当配列＝[0～3]: 10進数の各桁の値（絶対値）, [4]: スケール, [5]: 符号, [6]: 符号表示有無
    # 数値部は絶対値で保持しているので、そのまま比較すればよい
    # 戻り値: a > b の場合は 1, a < b の場合は -1, a == b の場合は 0
    for i in range(3, -1, -1):
        if a[i] > b[i]:
            return 1
        elif a[i] < b[i]:
            return -1
    return 0


@cuda.jit(cache=True)
def gpu_add(a, b, result):
    """
    加減算はスケール（小数点の位置）を合わせてから計算する必要があるので、最初にスケーリングを行う。
    """
    # 結果の小数点位置を取得
    result_scale = max(a[4], b[4])

    # a と b を結果の小数点位置に合わせてスケーリング
    scaled_a = cuda.local.array(7, dtype=numba.int32)
    scaled_b = cuda.local.array(7, dtype=numba.int32)

    # print("ADD-INPUT=", result[3], result[2], result[1], result[0])
    gpu_align_scale_larger(a, b, scaled_a, scaled_b)
    # print("ADD-SCALE=", result[3], result[2], result[1], result[0])
    for i in range(4):
        result[i] = 0

    if a[5] != b[5]:  # 符号が異なる場合は、減算を行う
        """
        減算の基本方針
        「絶対値の大きい方から小さい方を引く」
        理由：引き算は4つの数字全てを計算しないと符号が確定しない。
        それだと最後まで計算してから符号反転のために反転計算を行うことになって負荷が高いと思われる。
        そのため、まず絶対値で比較して、大きい方から小さい方を引くことで、最初から符号を確定させておく。
        これだと繰り下がり（borrow）が発生した際も、必ず上の桁の数字が存在する（マイナスにならない）ことが確定した状態で計算が進むため、
        最後に符号のフラグを反転するだけで済む。
        """
        borrow = 0
        compare = gpu_abs_compare(scaled_a, scaled_b)  # スケール調整後の絶対値で比較
        if compare > 0:
            result[5] = a[5]
            for i in range(4):
                # print("BEF-Calc=", i, scaled_a[i], scaled_b[i], borrow, result[3], result[2], result[1], result[0])
                temp = scaled_a[i] - scaled_b[i] - borrow
                # マイナスにならないように調整
                result[i] = (temp + 1_000_000_000) % 1_000_000_000
                borrow = 1 if temp < 0 else 0
                # print("AFT-Calc=", i, scaled_a[i], scaled_b[i], borrow, result[3], result[2], result[1], result[0])
        elif compare < 0:
            result[5] = b[5]
            for i in range(4):
                temp = scaled_b[i] - scaled_a[i] - borrow
                result[i] = (temp + 1_000_000_000) % 1_000_000_000
                borrow = 1 if temp < 0 else 0
        else:
            # 結果が 0 の場合は、符号を正に設定
            result[5] = 1
            for i in range(4):
                result[i] = 0

        # 減算はこの時点ではオーバーフローは考慮しない
        # ※絶対値が大きい方から小さい方を引いているため、桁が繰り上がることはない
    else:  # 符号が同じ場合は、加算を行う
        """
        加算の基本方針
        「桁ごとに足し算を行い、繰り上がりを次の桁に渡す」
        """
        carry = 0
        for i in range(4):
            temp = scaled_a[i] + scaled_b[i] + carry
            result[i] = temp % 1_000_000_000
            carry = temp // 1_000_000_000
        result[5] = a[5]

        # オーバーフローしていたら9で埋め尽くす
        if carry > 0:
            for i in range(4):
                result[i] = 999_999_999
    # print("ADD-CHECK=", result[3], result[2], result[1], result[0])

    # スケーリングと結果の格納
    scale_diff = result[4] - result_scale
    if scale_diff > 0:
        gpu_shift_left(result, scale_diff)
    elif scale_diff < 0:
        gpu_shift_right_round(result, -scale_diff)

    # print("ADD-AFTER=", result[3], result[2], result[1], result[0])


@cuda.jit(cache=True)
def gpu_sub(a, b, result):
    b[5] = -b[5]  # 計算用に符号を反転
    gpu_add(a, b, result)
    b[5] = -b[5]  # 元に戻す


@cuda.jit(cache=True)
def gpu_mul(a, b, result):
    # 結果の小数点位置を計算
    result_scale = a[4] + b[4]

    # 符号を計算
    result[5] = a[5] * b[5]

    # 展開方式 オーバーフローを抑えるためにint64で受け取る
    temp = cuda.local.array(4, dtype=numba.int64)
    temp[3] = a[3] * b[0] + a[2] * b[1] + a[1] * b[2] + a[0] * b[3]
    temp[2] = a[2] * b[0] + a[1] * b[1] + a[0] * b[2]
    temp[1] = a[1] * b[0] + a[0] * b[1]
    temp[0] = a[0] * b[0]

    # 結果の正規化と丸め
    carry = 0
    for i in range(4):
        temp[i] += carry
        result[i] = np.int32(temp[i] % 1_000_000_000)  # resultはint32なので型を合わせる
        carry = temp[i] // 1_000_000_000

    # オーバーフローしていたら9で埋め尽くす
    if carry > 0:
        for i in range(4):
            result[i] = 999_999_999

    # スケーリングと結果の格納
    scale_diff = result[4] - result_scale
    if scale_diff > 0:
        gpu_shift_left(result, scale_diff)
    elif scale_diff < 0:
        gpu_shift_right_round(result, -scale_diff)

    # 結果が 0 の場合は、符号を正に設定
    if gpu_is_zero(result):
        result[5] = 1


@cuda.jit(cache=True)
def gpu_div(a, b, result):
    """ GPU上で2つの10進数を除算します。 """
    # ゼロ除算のチェック
    if gpu_is_zero(b):
        # raise ZeroDivisionError("Division by zero")
        # print('ZeroDivisionError("Division by zero")')
        return
    c_s10 = cuda.const.array_like(cpu_c_s10)

    # 結果の小数点位置を計算
    result_scale = a[4] - b[4]

    # 符号を計算
    result[5] = a[5] * b[5]

    # 除算を行う際のワーク変数を定義
    dividend = cuda.local.array(7, dtype=numba.int32)
    divisor = cuda.local.array(7, dtype=numba.int32)
    for i in range(7):
        dividend[i] = a[i]
        divisor[i] = b[i]

    for i in range(4):
        result[i] = 0

    dividend_block_scale = gpu_get_scale_of_block(dividend)
    divisor_block_scale = gpu_get_scale_of_block(divisor)
    dividend_number_scale = gpu_get_scale_of_number(dividend, dividend_block_scale)
    divisor_number_scale = gpu_get_scale_of_number(divisor, divisor_block_scale)
    dividend_diff_scale = (3 - dividend_block_scale) * 9 + 9 - dividend_number_scale
    divisor_diff_scale = (3 - divisor_block_scale) * 9 + 9 - divisor_number_scale
    if dividend_block_scale == 0:
        dividend[3] = dividend[0]
        dividend[0] = 0
    elif dividend_block_scale == 1:
        dividend[3] = dividend[1]
        dividend[2] = dividend[0]
        dividend[1] = 0
        dividend[0] = 0
    elif dividend_block_scale == 2:
        dividend[3] = dividend[2]
        dividend[2] = dividend[1]
        dividend[1] = dividend[0]
        dividend[0] = 0

    if divisor_block_scale == 0:
        divisor[3] = divisor[0]
        divisor[0] = 0
    elif divisor_block_scale == 1:
        divisor[3] = divisor[1]
        divisor[2] = divisor[0]
        divisor[1] = 0
        divisor[0] = 0
    elif divisor_block_scale == 2:
        divisor[3] = divisor[2]
        divisor[2] = divisor[1]
        divisor[1] = divisor[0]
        divisor[0] = 0

    gpu_shift_left(dividend, 9 - dividend_number_scale)
    gpu_shift_left(divisor, 9 - divisor_number_scale)
    # dividendとdivisorの桁を揃えてから計算するので、結果に必要な小数桁数＋四捨五入用の追加桁＋桁移動分の差分ースケールの差分
    calc_scale = min(
        36, result[4] + 1 + divisor_diff_scale - dividend_diff_scale - result_scale + 1
    )
    # 四捨五入用に1桁多めに計算する必要があるので、いったんresultのscaleは0にして商を集計して最後に帳尻を合わせる。

    dividend[4] = 0
    divisor[4] = 0
    dividend[5] = 1
    divisor[5] = 1
    for scale in range(35 - 1, 36 - 1 - calc_scale, -1):
        # 引き算を行う
        current_block = scale // 9
        current_number = scale % 9
        for d in range(10):
            compare = gpu_abs_compare(dividend, divisor)
            if compare >= 0:
                # result(商）をインクリメント（正規化は後でやるので繰り上がりを無視して一旦雑にインクリメント）
                result[current_block] += c_s10[current_number]
                if compare == 0:
                    # 割り切れた（余りが0）ということなのでここで終了
                    for i in range(4):
                        dividend[i] = 0
                    break
                else:
                    # 引き算(gpu_subを呼ぶと判定済みの余計な処理が入ってしまって性能劣化するので最小限の引き算をする)
                    borrow = 0
                    for i in range(4):
                        temp = dividend[i] - divisor[i] - borrow
                        # マイナスにならないように調整
                        dividend[i] = (temp + 1_000_000_000) % 1_000_000_000
                        borrow = 1 if temp < 0 else 0
            else:
                # 繰り下げ
                break
        gpu_shift_right_round(divisor, 1)

        # # TODO 本当は余りがゼロになったらおしまいにしたいけど、そうすると桁合わせとかどうするかも直す必要があるので一旦手抜き
        # if gpu_is_zero(dividend):
        #     break

    # 結果の正規化と丸め
    carry = 0
    for i in range(4):
        temp = result[i] + carry
        result[i] = temp % 1_000_000_000
        carry = temp // 1_000_000_000

    # 四捨五入用の桁をシフトでつぶす（シフト関数の内部で四捨五入される）
    gpu_shift_right_round(result, 36 - calc_scale)

    # オーバーフローしていたら9で埋め尽くす（桁調整しているので起こらないと思ssう）
    if carry > 0:
        for i in range(4):
            result[i] = 999_999_999

    # 結果が 0 の場合は、符号を正に設定
    if gpu_is_zero(result):
        result[5] = 1


@cuda.jit(cache=True)
def gpu_d_move(fm, to):
    # 受け取り側のスケールに合わせるに必要がある
    # 元配列を壊さないためにscaledにコピーして処理する
    scaled = cuda.local.array(7, dtype=numba.int32)
    for i in range(7):
        scaled[i] = fm[i]

    # スケールが大きい方に合わせる
    if scaled[4] > to[4]:
        # 四捨五入でスケールダウン
        gpu_shift_right_round(scaled, scaled[4] - to[4])
    elif scaled[4] < to[4]:
        # スケールアップ
        gpu_shift_left(scaled, to[4] - scaled[4])
    else:
        # スケールが一致する場合は何もしない
        pass

    # スケール合わせが済んでから中身の値を転記
    to[0] = scaled[0]
    to[1] = scaled[1]
    to[2] = scaled[2]
    to[3] = scaled[3]
    # to[4] = fm[4] scale
    to[5] = scaled[5]  # sign
    # to[6] = fm[6] has_sign


@cuda.jit(cache=True)
def gpu_d_init(target: cupy.ndarray, value: cupy.int32, sign: cupy.int32):
    target[0] = value
    target[1] = 0
    target[2] = 0
    target[3] = 0
    target[5] = sign
    # target[4] = scale
    # target[6] = has_sign


@cuda.jit(cache=True)
def gpu_ge(a, b):
    a_scaled = cuda.local.array(7, dtype=numba.int32)
    b_scaled = cuda.local.array(7, dtype=numba.int32)
    gpu_align_scale_larger(a, b, a_scaled, b_scaled)
    if a_scaled[3] > b_scaled[3]:
        return True
    elif a_scaled[3] == b_scaled[3]:
        if a_scaled[2] > b_scaled[2]:
            return True
        elif a_scaled[2] == b_scaled[2]:
            if a_scaled[1] > b_scaled[1]:
                return True
            elif a_scaled[1] == b_scaled[1]:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


@cuda.jit(cache=True)
def gpu_le(a, b):
    a_scaled = cuda.local.array(7, dtype=numba.int32)
    b_scaled = cuda.local.array(7, dtype=numba.int32)
    gpu_align_scale_larger(a, b, a_scaled, b_scaled)
    if a_scaled[3] < b_scaled[3]:
        return True
    elif a_scaled[3] == b_scaled[3]:
        if a_scaled[2] < b_scaled[2]:
            return True
        elif a_scaled[2] == b_scaled[2]:
            if a_scaled[1] < b_scaled[1]:
                return True
            elif a_scaled[1] == b_scaled[1]:
                return True
            else:
                return False
        else:
            return False
    else:
        return False







