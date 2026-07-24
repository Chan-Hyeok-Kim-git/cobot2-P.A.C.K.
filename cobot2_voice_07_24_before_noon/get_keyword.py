# STT 텍스트에서 재난 유형을 찾아 반환하는 파일
class KeywordParser:
    # STT 문장에서 이 프로그램이 알아들을 수 있는 재난 이름을 찾는다.
    # 프로그램에서 처리할 수 있도록 미리 정한 재난 목록이다.
    ALLOWED_DISASTERS = {"지진", "홍수", "화재", "공습"}
    ALLOWED_ITEMS = {"장갑", "구급상자", "밧줄", "랜턴", "우비", "장화주머니", "방수포", "방독면", "호루라기", "보호안경", "비상식량", "비상 식량", "보호 안경", "장화 주머니", "구급 상자"}
    def extract_disaster(self, text):
        """
        STT 문장에서 허용된 재난 이름 하나를 찾아 반환.

        찾지 못하면 None 반환. 호출한 쪽은 None일 때 topic을 발행하지 않는다.
        """
        # STT 문장에 재난 이름이 포함되었는지 직접 검사.
        for disaster in self.ALLOWED_DISASTERS:
            # 예: "현재 재난 유형은 지진입니다" 안에는 "지진"이 들어 있음.
            if disaster in text:
                return disaster
        # 목록 어느 것도 없으면 호출자에게 실패를 알림.
        return None

    def extract_item(self, text):
        """
        STT 문장에서 허용된 물품 이름 하나를 찾아 반환.

        찾지 못하면 None 반환. 호출한 쪽은 None일 때 topic을 발행하지 않는다.
        """
        # STT 문장에 재난 이름이 포함되었는지 직접 검사.
        for item in self.ALLOWED_ITEMS:
            # 예: "현재 재난 유형은 지진입니다" 안에는 "지진"이 들어 있음.
            if item in text:
                return item
        # 목록 어느 것도 없으면 호출자에게 실패를 알림.
        return None
