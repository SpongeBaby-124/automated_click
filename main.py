"""
Web 自动化主程序 - 使用自然语言描述任务即可自动执行

使用方法：
    python main.py
    
然后输入自然语言任务描述，例如：
- "打开百度搜索Python教程并点击第一个结果"
- "在当前页面点击登录按钮"
"""

import asyncio
from dotenv import load_dotenv
from langgraph_app import create_automation_graph


async def execute_task(task_description: str):
    """
    执行自然语言描述的自动化任务
    
    Args:
        task_description: 任务的自然语言描述
        
    示例：
        await execute_task("打开谷歌搜索南京邮电大学官网")
        await execute_task("打开百度，搜索Python，点击第一个结果")
    """
    
    print("\n" + "=" * 70)
    print("🚀 Web 自动化任务执行器")
    print("=" * 70)
    print(f"\n📋 任务描述: {task_description}\n")
    
    print("⏳ 正在启动浏览器...\n")
    
    # 创建自动化图 (使用持久化上下文保存登录状态)
    graph, page, context, playwright = await create_automation_graph(
        headless=False  # 显示浏览器窗口
    )
    
    try:
        # 等待页面加载
        await asyncio.sleep(2)
        
        print("🤖 开始执行任务...\n")
        
        # 调用 graph.ainvoke() 执行任务
        result = await graph.ainvoke({
            "user_goal": task_description
        })
        
        # 打印执行过程
        print("\n" + "=" * 70)
        print("✅ 任务执行完成")
        print("=" * 70)
        
        print("\n📝 执行步骤:")
        for i, message in enumerate(result.get("messages", []), 1):
            content = message.content.strip()
            if content:
                print(f"  {i}. {content}")
        
        # 打印最终结果
        tool_result = result.get("tool_result")
        if tool_result:
            success = tool_result.get("success", False)
            message = tool_result.get("message", "")
            status = "✅ 成功" if success else "⚠️ 失败"
            print(f"\n🔧 最终执行结果: {status}")
            print(f"   {message}")
        
        print(f"\n🌐 当前页面: {page.url}")
        
        # 保持浏览器打开，等待用户手动关闭
        print("\n" + "=" * 70)
        print("💡 浏览器将保持打开状态")
        print("   - 你可以继续手动操作页面")
        print("   - 关闭浏览器标签页即可结束程序")
        print("=" * 70 + "\n")
        
        try:
            await page.wait_for_event("close", timeout=300000)  # 最多等待5分钟
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断任务")
    except Exception as e:
        print(f"\n\n❌ 任务执行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n🧹 正在清理资源...")
        await context.close()
        await playwright.stop()
        print("✅ 程序结束\n")


def main():
    """主函数 - 交互式输入任务"""
    
    # 加载环境变量
    load_dotenv()
    
    print("\n" + "=" * 70)
    print("🤖 LangGraph Web 自动化助手")
    print("=" * 70)
    print("\n使用自然语言描述你想执行的任务，例如：")
    print("  - 打开谷歌搜索南京邮电大学官网")
    print("  - 打开百度搜索Python教程并点击第一个结果")
    print("  - 在淘宝搜索iPhone 15")
    print("\n💾 登录状态保存：")
    print("  - 浏览器数据保存在 ./browser_data 目录")
    print("  - 登录信息、Cookies 将自动保留")
    print("  - 下次运行时无需重复登录")
    print("\n提示：输入 'quit' 或 'exit' 退出程序")
    print("=" * 70 + "\n")
    
    while True:
        try:
            # 获取用户输入
            task = input("📝 请输入任务描述: ").strip()
            
            # 检查退出命令
            if task.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n👋 再见！")
                break
            
            # 检查空输入
            if not task:
                print("⚠️ 任务描述不能为空，请重新输入\n")
                continue
            
            # 执行任务
            asyncio.run(execute_task(task))
            
            # 询问是否继续
            print("\n" + "=" * 70)
            continue_choice = input("是否继续执行新任务？(y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '是', '']:
                print("\n👋 再见！")
                break
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
