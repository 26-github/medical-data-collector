from fastmcp import FastMCP
import sys
import logging
from datetime import datetime, timedelta
import requests
import json

# 设置详细的日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("FastMCP-Server")

# 创建MCP服务器
mcp = FastMCP(
    name="medical_assistant_tools",
    instructions="医疗助手工具服务器,能查询医院信息，提供日期、时间等辅助功能",
)

# MCP工具
@mcp.tool()
def get_current_date() -> str:
    """获取当前日期"""
    result = datetime.now().strftime("%Y年%m月%d日")
    logger.info(f"get_current_date called, returning: {result}")
    return result


@mcp.tool()
def get_current_time() -> str:
    """获取当前时间"""
    result = datetime.now().strftime("%H:%M:%S")
    logger.info(f"get_current_time called, returning: {result}")
    return result


@mcp.tool()
def get_hospital_info(cnName: str = "", address: str = "", enName: str = "", cnShort: str = "", hospitalType: str = "", level: str = "", ownership: str = "", pageSize: int = 10) -> str:
    """获取医院信息
    
    Args:
        cnName: 医院中文名称（可选，用于搜索特定医院）
        address: 医院地址（可选，用于按地区搜索医院）
        enName: 医院英文名称（可选）
        cnShort: 医院中文简称（可选）
        hospitalType: 医院类型（可选，如：专科医院、综合性医院、中医医院）
        level: 医院等级（可选，如：三级甲等、二级甲等）
        ownership: 医院性质（可选，如：公立、私立）
        pageSize: 每页显示数量，默认10条
    """
    try:
        # API URL
        url = "http://localhost:48080/admin-api/datamanagement/hospital/page"
        
        # 请求参数
        params = {
            "pageSize": pageSize
        }
        
        # 添加所有搜索条件到参数中
        if cnName:
            params["cnName"] = cnName
        if address:
            params["address"] = address
        if enName:
            params["enName"] = enName
        if cnShort:
            params["cnShort"] = cnShort
        if hospitalType:
            params["hospitalType"] = hospitalType
        if level:
            params["level"] = level
        if ownership:
            params["ownership"] = ownership
        
        # 请求头
        headers = {
            "tenant-id": "1",
            "Content-Type": "application/json"
        }
        
        logger.info(f"get_hospital_info called with cnName='{cnName}', address='{address}', enName='{enName}', cnShort='{cnShort}', hospitalType='{hospitalType}', level='{level}', ownership='{ownership}', pageSize={pageSize}")
        
        # 发送HTTP请求
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 解析响应数据
        data = response.json()
        
        if data.get("code") == 0:
            hospital_list = data.get("data", {}).get("list", [])
            total = data.get("data", {}).get("total", 0)
            
            if hospital_list:
                # 格式化医院信息
                search_conditions = []
                if cnName:
                    search_conditions.append(f"医院名称: {cnName}")
                if address:
                    search_conditions.append(f"地址: {address}")
                if enName:
                    search_conditions.append(f"英文名称: {enName}")
                if cnShort:
                    search_conditions.append(f"中文简称: {cnShort}")
                if hospitalType:
                    search_conditions.append(f"医院类型: {hospitalType}")
                if level:
                    search_conditions.append(f"医院等级: {level}")
                if ownership:
                    search_conditions.append(f"医院性质: {ownership}")
                
                result_info = f"找到 {total} 家医院"
                if search_conditions:
                    search_info = "、".join(search_conditions)
                    result_info += f"（搜索条件：{search_info}）"
                result_info += "：\n\n"
                
                for hospital in hospital_list:
                    result_info += f"医院中文名称：{hospital.get('cnName', '未知')}\n"
                    result_info += f"医院中文缩写名称：{hospital.get('cnShort', '未知')}\n"
                    result_info += f"医院英文名称：{hospital.get('enName', '未知')}\n"
                    result_info += f"医院类型：{hospital.get('hospitalType', '未知')}\n"
                    result_info += f"医院等级：{hospital.get('level', '未知')}\n"
                    result_info += f"所有制：{hospital.get('ownership', '未知')}\n"
                    result_info += f"地址：{hospital.get('address', '未知')}\n"
                    result_info += f"电话：{hospital.get('phone', '未提供')}\n"
                    result_info += f"床位数：{hospital.get('bedCount', 0)}\n"
                    result_info += f"员工数：{hospital.get('staffCount', 0)}\n"
                    if hospital.get('establishedYear'):
                        result_info += f"建院时间：{hospital.get('establishedYear')}年\n"
                    if hospital.get('introduction'):
                        intro = hospital.get('introduction', '')
                        result_info += f"医院简介：{intro}...\n"
                    result_info += f"网站：{hospital.get('website', '未提供')}\n"
                    result_info += "\n" + "="*50 + "\n\n"
                
                logger.info(f"get_hospital_info success: found {total} hospitals")
                return result_info
            else:
                search_conditions = []
                if cnName:
                    search_conditions.append(f"医院名称: {cnName}")
                if address:
                    search_conditions.append(f"地址: {address}")
                if enName:
                    search_conditions.append(f"英文名称: {enName}")
                if cnShort:
                    search_conditions.append(f"中文简称: {cnShort}")
                if hospitalType:
                    search_conditions.append(f"医院类型: {hospitalType}")
                if level:
                    search_conditions.append(f"医院等级: {level}")
                if ownership:
                    search_conditions.append(f"医院性质: {ownership}")
                search_text = "、".join(search_conditions) if search_conditions else "无特定条件"
                
                result = f"未找到符合条件的医院（搜索条件：{search_text}）"
                logger.info(f"get_hospital_info: no hospitals found")
                return result
        else:
            error_msg = f"API返回错误：{data.get('msg', '未知错误')}"
            logger.error(f"get_hospital_info API error: {error_msg}")
            return error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"网络请求失败：{str(e)}"
        logger.error(f"get_hospital_info network error: {error_msg}")
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析失败：{str(e)}"
        logger.error(f"get_hospital_info JSON error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"获取医院信息时发生错误：{str(e)}"
        logger.error(f"get_hospital_info unexpected error: {error_msg}")
        return error_msg

# 启动MCP服务器
if __name__ == "__main__":

    # 启动服务器
    mcp.run(
        transport="sse",
        host="127.0.0.1",
        port=8001
    )
