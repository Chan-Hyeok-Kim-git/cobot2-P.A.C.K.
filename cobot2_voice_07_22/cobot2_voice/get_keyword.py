"""STT 텍스트에서 허용된 도구와 위치를 추출한다."""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class KeywordParser:
    ALLOWED_DISASTERS = {"지진", "홍수", "화재", "공습"}

    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 사용자의 문장에서 특정 재난을 추출해야 합니다.

            <목표>
            - 문장에서 다음 상황에 포함된 재난을 최대한 정확히 추출하세요.

            <재난 리스트>
            - 지진, 홍수, 화재, 공습

            <출력 형식>
            - 다음 형식을 반드시 따르세요: [재난]
            - 재난은 단 하나

            <예시>
            - 입력: "현재 재난 유형은 'OO'입니다. 안내에 따라 비상용품을 준비하십시오. 다시 알려드립니다. 현재 재난 유형은 'OO'입니다."  
            출력: OO

            <사용자 입력>
            "{user_input}"                
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm

    def extract(self, text):
        response = self.lang_chain.invoke({"user_input": text})
        disaster = response.content.strip()

        # LLM 출력이 허용 목록의 정확한 값일 때만 전달한다.
        for disaster in self.ALLOWED_DISASTERS:
            if disaster in text:
                return disaster
        return None
